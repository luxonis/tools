import sys

sys.path.append("./tools/yolo/yolov5")
import onnx
import onnxsim
import torch.nn as nn
from models.common import Conv
from models.experimental import attempt_load
from models.yolo import Detect
from utils.activations import SiLU
import torch
from typing import Tuple

from tools.modules import Exporter, DetectV5


class YoloV5Exporter(Exporter):
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
            subtype="yolov5",
            output_names=["output1_yolov5", "output2_yolov5", "output3_yolov5"],
        )
        self.load_model()

    def load_model(self):
        # code based on export.py from YoloV5 repository
        # load the model
        model = attempt_load(self.model_path, device="cpu")  # load FP32 model

        # check num classes and labels
        assert model.nc == len(
            model.names
        ), f"Model class count {model.nc} != len(names) {len(model.names)}"

        # check if image size is suitable
        gs = int(max(model.stride))  # grid size (max stride)
        if isinstance(self.imgsz, int):
            self.imgsz = [self.imgsz, self.imgsz]
        for sz in self.imgsz:
            if sz % gs != 0:
                raise ValueError(f"Image size is not a multiple of maximum stride {gs}")

        # ensure correct length
        if len(self.imgsz) != 2:
            raise ValueError("Image size must be of length 1 or 2.")

        inplace = True

        for k, m in model.named_modules():
            if isinstance(m, Conv):  # assign export-friendly activations
                if isinstance(m.act, nn.SiLU):
                    m.act = SiLU()
            elif isinstance(m, Detect):
                m.inplace = inplace
                m.onnx_dynamic = False
                if hasattr(m, "forward_export"):
                    m.forward = m.forward_export  # assign custom forward (optional)

        if hasattr(model, "module"):
            model.module.model[-1] = DetectV5(model.module.model[-1])
        else:
            model.model[-1] = DetectV5(model.model[-1])

        model.eval()
        self.model = model

        m = model.module.model[-1] if hasattr(model, "module") else model.model[-1]
        self.num_branches = len(m.anchor_grid)

    def export_nn_archive(self):
        """Export the model to NN archive format."""
        self.make_nn_archive(list(self.model.names.values()), self.model.nc)
