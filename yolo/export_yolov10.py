import sys
sys.path.append("./yolo/ultralytics")

import re
import torch
import onnxsim
import onnx
from pathlib import Path


from exporter import Exporter
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.nn.modules import Detect
from yolo.detect_head import DetectV10
from yolo.export_yolov8 import YoloV8Exporter


DIR_TMP = "./tmp"

class YoloV10Exporter(YoloV8Exporter):
    
    def load_model(self):
        # load the model
        # model = attempt_load_weights(str(self.weights_path.resolve()), device="cpu", inplace=True, fuse=True)
        model, _ = attempt_load_one_weight(str(self.weights_path.resolve()), device="cpu", inplace=True, fuse=True)

        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        
        # check num classes and labels
        # assert model.nc == len(names), f'Model class count {model.nc} != len(names) {len(names)}'
        assert model.yaml["nc"] == len(names), f'Model class count {model.yaml["nc"]} != len(names) {len(names)}'

        # Replace with the custom Detection Head
        if isinstance(model.model[-1], (Detect)):
            model.model[-1] = DetectV10(model.model[-1], self.use_rvc2)

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
    

if __name__ == "__main__":
    # Test the YoloV10Exporter
    conv_path = Path(DIR_TMP)
    weights_filename = "yolov10n.pt"
    imgsz = 640
    conv_id = "test"
    nShaves = 6
    useLegacyFrontend = 'false'
    useRVC2 = 'false'
    exporter = YoloV10Exporter(conv_path, weights_filename, imgsz, conv_id, nShaves, useLegacyFrontend, useRVC2)
    exporter.export_onnx()
    exporter.export_openvino("v10")
    exporter.export_json()
