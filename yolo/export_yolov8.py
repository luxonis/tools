import sys
sys.path.append("./yolo/ultralytics")

import torch
import onnxsim
import onnx

from exporter import Exporter
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.nn.modules import Detect
from yolo.detect_head import DetectV8


DIR_TMP = "./tmp"

class YoloV8Exporter(Exporter):

    def __init__(self, conv_path, weights_filename, imgsz, conv_id, n_shaves=6):
        super().__init__(conv_path, weights_filename, imgsz, conv_id, n_shaves)
        self.load_model()
    
    def load_model(self):
        # load the model
        model = attempt_load_weights(str(self.weights_path.resolve()), device="cpu", inplace=True, fuse=True)

        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        
        # check num classes and labels
        assert model.nc == len(names), f'Model class count {model.nc} != len(names) {len(names)}'

        # Replace with the custom Detection Head
        if isinstance(model.model[-1], (Detect)):
            model.model[-1] = DetectV8(model.model[-1])

        self.num_branches = model.model[-1].nl

        # check if image size is suitable
        gs = max(int(model.stride.max()), 32)  # model stride
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

        nc = self.model.model[-1].nc
        names = [f"Class_{i}" for i in range(nc)]

        return self.write_json(anchors, masks, nc, names)
