import sys

sys.path.append("./tools/yolov7/yolov7")

from models.experimental import attempt_load
from models.common import Conv
from models.yolo import Detect
from typing import Tuple
import onnx
import onnxsim
import torch

from tools.modules import Exporter, DetectV7


class YoloV7Exporter(Exporter):
    def __init__(
        self,
        model_path: str,
        imgsz: Tuple[int, int],
        use_rvc2: bool,
    ):
        super().__init__(
            model_path,
            imgsz,
            use_rvc2,
            subtype="yolov7",
            output_names=["output1_yolov7", "output2_yolov7", "output3_yolov7"],
        )
        self.load_model()
    
    def load_model(self):
        # code based on export.py from YoloV5 repository
        # load the model
        model = attempt_load(self.model_path, map_location="cpu")
        # check num classes and labels
        assert model.nc == len(model.names), f'Model class count {model.nc} != len(names) {len(model.names)}'

        if hasattr(model, "module"):
            model.module.model[-1] = DetectV7(model.module.model[-1])
        else:
            model.model[-1] = DetectV7(model.model[-1])
        
        # check if image size is suitable
        gs = int(max(model.stride))  # grid size (max stride)
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

        m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
        self.num_branches = len(m.anchor_grid)           

    def export_nn_archive(self):
        """Export the model to NN archive format."""
        self.make_nn_archive(self.model.names, self.model.nc)
