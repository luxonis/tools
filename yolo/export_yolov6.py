import sys
sys.path.append("./yolo/YOLOv6")

import torch
from yolov6.models.heads.effidehead_distill_ns import Detect
from yolov6.layers.common import RepVGGBlock
import yolov6.utils.checkpoint
import onnx
from exporter import Exporter

import onnxsim

from yolo.detect_head import DetectV6R4s, DetectV6R4m
from yolo.backbones import YoloV6BackBone


# Override with your custom implementation
def load_checkpoint(weights, map_location=None, inplace=True, fuse=True):
  """Load model from checkpoint file with weights only set to `False`."""
  from yolov6.utils.events import LOGGER
  from yolov6.utils.torch_utils import fuse_model

  LOGGER.info("Loading checkpoint from {}".format(weights))
  ckpt = torch.load(weights, map_location=map_location, weights_only=False)  # load
  model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
  if fuse:
      LOGGER.info("\nFusing model...")
      model = fuse_model(model).eval()
  else:
      model = model.eval()
  return model


# Replace the original function
yolov6.utils.checkpoint.load_checkpoint = load_checkpoint


class YoloV6R4Exporter(Exporter):

    def __init__(self, conv_path, weights_filename, imgsz, conv_id, n_shaves=6, use_legacy_frontend='false', use_rvc2='true'):
        super().__init__(conv_path, weights_filename, imgsz, conv_id, n_shaves, use_legacy_frontend, use_rvc2)
        self.load_model()
    
    def load_model(self):

        # code based on export.py from YoloV5 repository
        # load the model
        model = load_checkpoint(str(self.weights_path.resolve()), map_location="cpu", inplace=True, fuse=True)  # load FP32 model
        
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()

        if isinstance(model.detect, Detect):
            model.detect = DetectV6R4s(model.detect, self.use_rvc2)
        else:
            model.detect = DetectV6R4m(model.detect, self.use_rvc2)
        
        self.num_branches = len(model.detect.grid)

        # check if image size is suitable
        gs = 2 ** (2 + self.num_branches)  # 1 = 8, 2 = 16, 3 = 32
        if isinstance(self.imgsz, int):
            self.imgsz = [self.imgsz, self.imgsz]
        for sz in self.imgsz:
            if sz % gs != 0:
                raise ValueError(f"Image size is not a multiple of maximum stride {gs}")

        # ensure correct length
        if len(self.imgsz) != 2:
            raise ValueError(f"Image size must be of length 1 or 2.")

        model.eval()
        self.model = model

    def export_onnx(self):
        # export onnx model
        self.f_onnx = (self.conv_path / f"{self.model_name}.onnx").resolve()
        im = torch.zeros(1, 3, *self.imgsz[::-1])#.to(device)  # image size(1,3,320,192) BCHW iDetection
        torch.onnx.export(self.model, im, self.f_onnx, verbose=False, opset_version=12,
                        training=torch.onnx.TrainingMode.EVAL,
                        do_constant_folding=True,
                        input_names=['images'],
                        output_names=[f"output{i+1}_yolov6r2" for i in range(self.num_branches)],
                        dynamic_axes=None)

        # check if the arhcitecture is correct
        model_onnx = onnx.load(self.f_onnx)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # simplify the moodel

        onnx_model, check = onnxsim.simplify(model_onnx)
        assert check, 'assert check failed'
    
        onnx.checker.check_model(onnx_model)  # check onnx model

        # save the simplified model
        self.f_simplified = (self.conv_path / f"{self.model_name}-simplified.onnx").resolve()
        onnx.save(onnx_model, self.f_simplified)
        return self.f_simplified
    
    def export_openvino(self, version):
        return super().export_openvino('v6r2')

    def export_json(self):
        # generate anchors and sides
        anchors, masks = [], {}

        nc = self.model.detect.nc
        names = [f"Class_{i}" for i in range(nc)]

        return self.write_json(anchors, masks, nc, names)
