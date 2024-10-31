from __future__ import annotations

import sys
from typing import Tuple

from tools.modules import DetectV6R1, Exporter
from tools.utils import get_first_conv2d_in_channels

sys.path.append("./tools/yolov6r1/YOLOv6R1")
from yolov6.layers.common import RepVGGBlock  # noqa: E402
from yolov6.utils.checkpoint import load_checkpoint  # noqa: E402


class YoloV6R1Exporter(Exporter):
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
            subtype="yolov6",
            output_names=["output1_yolov6", "output2_yolov6", "output3_yolov6"],
        )
        self.load_model()

    def load_model(self):
        # code based on export.py from YoloV5 repository
        # load the model
        model = load_checkpoint(
            self.model_path, map_location="cpu", inplace=True, fuse=True
        )  # load FP32 model

        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()

        if hasattr(model.detect, "obj_preds"):
            model.detect = DetectV6R1(model.detect)
        else:
            raise ValueError(
                "Error while loading model (This may be caused by trying to convert either the latest release 4.0 that isn't supported yet, or by releases 2.0 or 3.0, in which case, try to convert using the 'YoloV6 (R2, R3)' option)."
            )

        self.num_branches = len(model.detect.grid)

        try:
            self.number_of_channels = get_first_conv2d_in_channels(model)
            # print(f"Number of channels: {self.number_of_channels}")
        except Exception as e:
            print(f"Error while getting number of channels: {e}")

        # check if image size is suitable
        gs = 2 ** (2 + self.num_branches)  # 1 = 8, 2 = 16, 3 = 32
        if isinstance(self.imgsz, int):
            self.imgsz = [self.imgsz, self.imgsz]
        for sz in self.imgsz:
            if sz % gs != 0:
                raise ValueError(f"Image size is not a multiple of maximum stride {gs}")

        # ensure correct length
        if len(self.imgsz) != 2:
            raise ValueError("Image size must be of length 1 or 2.")

        model.eval()
        self.model = model
