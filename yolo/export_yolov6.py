import sys
sys.path.append("./yolo/YOLOv6")

import torch
from yolov6.layers.common import RepVGGBlock
from yolov6.utils.checkpoint import load_checkpoint
import onnx
from exporter import Exporter

DIR_TMP = "./tmp"

class YoloV6Exporter(Exporter):

    def __init__(self, conv_path, weights_filename, imgsz, conv_id):
        super().__init__(conv_path, weights_filename, imgsz, conv_id)
        self.load_model()
        

    
    def load_model(self):

        # code based on export.py from YoloV5 repository
        # load the model
        #model = DetectBackend(str(self.weights_path.resolve()), device="cpu")
        model = load_checkpoint(str(self.weights_path.resolve()), map_location="cpu", inplace=True, fuse=True)  # load FP32 model
        for layer in model.modules():
            #print(type(layer))
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()

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
        
        # fuse RepVGG blocks
        #for layer in model.modules():
        #    print(type(layer))
        #    if isinstance(layer, RepVGGBlock):
        #        layer.switch_to_deploy()

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

        #output_names = []
        for i, idx in enumerate(outputs):
            onnx_model.graph.node[idx].name = f"output{i+1}_yolov6"
            #output_names.append(onnx_model.graph.node[idx].name)

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