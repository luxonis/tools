import sys

from yolo.detect_head import DetectV1
sys.path.append("./yolo/YOLOv6R1")

import torch
from yolov6.layers.common import RepVGGBlock
from yolov6.utils.checkpoint import load_checkpoint
import onnx
from exporter import Exporter

DIR_TMP = "./tmp"

class YoloV6R1Exporter(Exporter):

    def __init__(self, conv_path, weights_filename, imgsz, conv_id, n_shaves=6, use_legacy_frontend='false'):
        super().__init__(conv_path, weights_filename, imgsz, conv_id, n_shaves, use_legacy_frontend)
        self.load_model()
    
    def load_model(self):
        # code based on export.py from YoloV5 repository
        # load the model
        model = load_checkpoint(str(self.weights_path.resolve()), map_location="cpu", inplace=True, fuse=True)  # load FP32 model
        
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
        
        if hasattr(model.detect, 'obj_preds'):
            model.detect = DetectV1(model.detect)
        else:
            raise ValueError(f"Error while loading model (This may be caused by trying to convert a newer version of YoloV6 - release 2.0 or 3.0, if that is the case, try to convert using the `YoloV6 (R2, R3)` option, or by trying to convert the latest release 4.0 that isn't supported yet).")
        
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
        onnx_model, check = self.get_onnx()
        assert check, 'assert check failed'
        # get contacts ready for parsing
        conc_idx = []
        for i, n in enumerate(onnx_model.graph.node):
            if "Concat" in n.name:
                conc_idx.append(i)

        outputs = conc_idx[-(self.num_branches+1):-1]
        change_inputs = []

        for i, idx in enumerate(outputs):
            onnx_model.graph.node[idx].name = f"output{i+1}_yolov6"
            change_inputs.append(onnx_model.graph.node[idx].output[0])
            onnx_model.graph.node[idx].output[0] = f"output{i+1}_yolov6"
        
        for i in range(len(onnx_model.graph.node)):
            for j in range(len(onnx_model.graph.node[i].input)):
                if onnx_model.graph.node[i].input[j] in change_inputs:
                    output_i = change_inputs.index(onnx_model.graph.node[i].input[j])+1
                    onnx_model.graph.node[i].input[j] = f"output{output_i}_yolov6"
            
        onnx.checker.check_model(onnx_model)  # check onnx model

        # save the simplified model
        self.f_simplified = (self.conv_path / f"{self.model_name}-simplified.onnx").resolve()
        onnx.save(onnx_model, self.f_simplified)
        return self.f_simplified

    def export_json(self):
        # generate anchors and sides
        anchors, masks = [], {}

        nc = self.model.detect.nc
        names = [f"Class_{i}" for i in range(nc)]

        return self.write_json(anchors, masks, nc, names)

