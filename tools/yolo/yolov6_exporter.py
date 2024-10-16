from __future__ import annotations

import sys
from typing import Tuple

from tools.modules import DetectV6R4m, DetectV6R4s, Exporter
from tools.utils import get_first_conv2d_in_channels

sys.path.append("./tools/yolo/YOLOv6")
from yolov6.layers.common import RepVGGBlock  # noqa: E402
from yolov6.models.heads.effidehead_distill_ns import Detect  # noqa: E402
from yolov6.utils.checkpoint import load_checkpoint  # noqa: E402


class YoloV6R4Exporter(Exporter):
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
            output_names=["output1_yolov6r2", "output2_yolov6r2", "output3_yolov6r2"],
        )
        self.load_model()

    def load_model(self):
        # code based on export.py from YoloV5 repository
        # load the model
        model = load_checkpoint(
            self.model_path,
            map_location="cpu",
            inplace=True,
            fuse=True,
        )  # load FP32 model

        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()

        if isinstance(model.detect, Detect):
            model.detect = DetectV6R4s(model.detect, self.use_rvc2)
        else:
            model.detect = DetectV6R4m(model.detect, self.use_rvc2)

        try:
            self.number_of_channels = get_first_conv2d_in_channels(model)
            # print(f"Number of channels: {self.number_of_channels}")
        except Exception as e:
            print(f"Error while getting number of channels: {e}")

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
            raise ValueError("Image size must be of length 1 or 2.")

        model.eval()
        self.model = model
