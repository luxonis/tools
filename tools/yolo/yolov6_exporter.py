from __future__ import annotations

import os
import sys
from typing import Tuple

import torch
from loguru import logger

from tools.modules import DetectV6R4m, DetectV6R4s, Exporter
from tools.utils import get_first_conv2d_in_channels

current_dir = os.path.dirname(os.path.abspath(__file__))
yolov6_path = os.path.join(current_dir, "YOLOv6")
sys.path.append(yolov6_path)

import yolov6.utils.checkpoint  # noqa: E402
from yolov6.layers.common import RepVGGBlock  # noqa: E402
from yolov6.models.heads.effidehead_distill_ns import Detect  # noqa: E402


# Override with your custom implementation
def load_checkpoint(weights, map_location=None, inplace=True, fuse=True):
    """Load model from checkpoint file."""
    from yolov6.utils.events import LOGGER  # noqa: E402
    from yolov6.utils.torch_utils import fuse_model  # noqa: E402

    LOGGER.info("Loading checkpoint from {}".format(weights))
    ckpt = torch.load(weights, map_location=map_location, weights_only=False)  # load
    model = ckpt["ema" if ckpt.get("ema") else "model"].float()
    if fuse:
        LOGGER.info("\nFusing model...")
        model = fuse_model(model).eval()
    else:
        model = model.eval()
    return model


# Replace the original function
yolov6.utils.checkpoint.load_checkpoint = load_checkpoint


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
            subtype="yolov6r2",
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
            logger.error(f"Error while getting number of channels: {e}")

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
