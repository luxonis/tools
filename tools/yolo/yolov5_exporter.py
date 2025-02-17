from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import torch.nn as nn
from loguru import logger

from tools.modules import DetectV5, Exporter
from tools.utils import get_first_conv2d_in_channels

current_dir = os.path.dirname(os.path.abspath(__file__))
yolov5_path = os.path.join(current_dir, "yolov5")
sys.path.append(yolov5_path)

from models.common import Conv  # noqa: E402
from models.experimental import attempt_load  # noqa: E402
from models.yolo import Detect  # noqa: E402
from utils.activations import SiLU  # noqa: E402


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

        for _, m in model.named_modules():
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

        try:
            self.number_of_channels = get_first_conv2d_in_channels(model)
            # print(f"Number of channels: {self.number_of_channels}")
        except Exception as e:
            logger.error(f"Error while getting number of channels: {e}")

        self.m = model.module.model[-1] if hasattr(model, "module") else model.model[-1]
        self.num_branches = len(self.m.anchor_grid)

    def export_nn_archive(self, class_names: Optional[List[str]] = None):
        """
        Export the model to NN archive format.

        Args:
            class_list (Optional[List[str]], optional): List of class names. Defaults to None.
        """
        names = list(self.model.names.values())

        if class_names is not None:
            assert (
                len(class_names) == self.model.nc
            ), f"Number of the given class names {len(class_names)} does not match number of classes {self.model.nc} provided in the model!"
            names = class_names

        anchors = [
            self.m.anchor_grid[i][0, :, 0, 0].numpy().tolist()
            for i in range(self.num_branches)
        ]
        self.make_nn_archive(
            names, self.model.nc, parser="YOLOExtendedParser", anchors=anchors
        )
