from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger

from tools.modules import DetectV5, Exporter
from tools.utils import get_first_conv2d_in_channels, patch_pathlib_for_cross_platform
from tools.utils.constants import Encoding

current_dir = os.path.dirname(os.path.abspath(__file__))
yolov5_path = os.path.join(current_dir, "yolov5")
# Ensure it's first in sys.path
if yolov5_path not in sys.path:
    sys.path.insert(0, yolov5_path)


import models.experimental  # noqa: E402
from models.common import Conv  # noqa: E402
from models.yolo import Detect as DetectYOLOv5  # noqa: E402
from utils.activations import SiLU  # noqa: E402


def attempt_load_yolov5(weights, device=None, inplace=True, fuse=True):
    patch_pathlib_for_cross_platform()
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    from models.yolo import Detect, Model  # noqa: E402

    model = models.experimental.Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(
            models.experimental.attempt_download(w),
            map_location="cpu",
            weights_only=False,
        )  # load
        ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, "stride"):
            ckpt.stride = torch.tensor([32.0])
        if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(
            ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval()
        )  # model in eval mode

    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace  # torch 1.7.0 compatibility
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")
                m.anchor_grid = [torch.zeros(1)] * m.nl
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))
    model.stride = model[
        torch.argmax(torch.tensor([m.stride.max() for m in model])).int()
    ].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), (
        f"Models have different class counts: {[m.nc for m in model]}"
    )
    return model


# Replace the original function
models.experimental.attempt_load = attempt_load_yolov5


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
        model = attempt_load_yolov5(self.model_path, device="cpu")  # load FP32 model

        # check num classes and labels
        assert model.nc == len(model.names), (
            f"Model class count {model.nc} != len(names) {len(model.names)}"
        )

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
            elif isinstance(m, DetectYOLOv5):
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

    def export_nn_archive(
        self, class_names: Optional[List[str]] = None, encoding: Encoding = Encoding.RGB
    ):
        """Export the model to NN archive format.

        Args:
            class_list (Optional[List[str]], optional): List of class names. Defaults to None.
            encoding (Encoding): Color encoding used in the input model. Defaults to RGB.
        """
        names = list(self.model.names.values())

        if class_names is not None:
            assert len(class_names) == self.model.nc, (
                f"Number of the given class names {len(class_names)} does not match number of classes {self.model.nc} provided in the model!"
            )
            names = class_names

        anchors = [
            self.m.anchor_grid[i][0, :, 0, 0].numpy().tolist()
            for i in range(self.num_branches)
        ]
        self.make_nn_archive(
            class_list=names,
            n_classes=self.model.nc,
            parser="YOLOExtendedParser",
            anchors=anchors,
            encoding=encoding,
        )
