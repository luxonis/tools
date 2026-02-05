from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

from loguru import logger

from tools.modules import DetectV26, Exporter, PoseV26, SegmentV26
from tools.utils import get_first_conv2d_in_channels
from tools.utils.constants import Encoding

current_dir = os.path.dirname(os.path.abspath(__file__))
yolo_path = os.path.join(current_dir, "ultralytics")
sys.path.append(yolo_path)

from ultralytics.nn.modules import (  # noqa: E402
    Detect,
    Pose26,
    Segment26,
)
from ultralytics.nn.tasks import load_checkpoint  # noqa: E402

DETECT_MODE = 0
SEGMENT_MODE = 1
OBB_MODE = 2
CLASSIFY_MODE = 3
POSE_MODE = 4


def get_output_names(mode: int):
    if mode == DETECT_MODE:
        return ["output"]
    elif mode == SEGMENT_MODE:
        return ["output", "mask_output", "protos_output"]
    elif mode == POSE_MODE:
        return ["output", "kpt_output"]
    else:
        logger.warning("Unsupported task type for YOLO26, conversion may fail")
        return ["output"]


def get_yolo_output_names(mode: int = 0):
    # For now, yolo output names doesn't differ based on mode because we no longer extract 3 outputs from FPN
    return ["output"]


class Yolo26Exporter(Exporter):
    def __init__(self, model_path: str, imgsz: Tuple[int, int], use_rvc2: bool):
        super().__init__(
            model_path,
            imgsz,
            use_rvc2,
            subtype="yolo26",
            output_names=["output"],
        )
        self.load_model()

    def load_model(self):
        model, _ = load_checkpoint(
            self.model_path, device="cpu", inplace=True, fuse=False
        )

        head = model.model[-1]
        self.mode = -1
        if isinstance(head, (Segment26)):
            model.model[-1] = SegmentV26(model.model[-1], self.use_rvc2)
            self.mode = SEGMENT_MODE
        elif isinstance(head, Pose26):
            model.model[-1] = PoseV26(model.model[-1], self.use_rvc2)
            self.mode = POSE_MODE
        elif isinstance(head, Detect):
            model.model[-1] = DetectV26(head, self.use_rvc2)
            self.mode = DETECT_MODE

        if self.mode in [DETECT_MODE, SEGMENT_MODE, POSE_MODE]:
            self.names = (
                model.module.names if hasattr(model, "module") else model.names
            )  # get class names
            # check num classes and labels
            assert model.model[-1].nc == len(self.names), (
                f"Model class count {model.nc} != len(names) {len(self.names)}"
            )

        try:
            self.number_of_channels = get_first_conv2d_in_channels(model)
        except Exception as e:
            logger.error(f"Error while getting number of channels: {e}")

        self.all_output_names = get_output_names(self.mode)
        self.output_names = get_yolo_output_names(self.mode)

        gs = max(int(model.stride.max()), 32)  # model stride
        if isinstance(self.imgsz, int):
            self.imgsz = [self.imgsz, self.imgsz]
        for sz in self.imgsz:
            if sz % gs != 0:
                raise ValueError(f"Image size is not a multiple of maximum stride {gs}")

        # ensure correct length
        if len(self.imgsz) != 2:
            raise ValueError("Image size must be of length 1 or 2.")

        model.fuse()
        model.eval()
        self.model = model

    def export_nn_archive(
        self, class_names: Optional[List[str]] = None, encoding: Encoding = Encoding.RGB
    ):
        names = list(self.model.names.values())

        if class_names is not None:
            assert len(class_names) == len(names), (
                f"Number of the given class names {len(class_names)} does not match number of classes {len(names)} provided in the model!"
            )
            names = class_names

        self.f_nn_archive = (self.output_folder / f"{self.model_name}.tar.xz").resolve()

        if self.mode == DETECT_MODE:
            self.make_nn_archive(
                class_list=names, n_classes=self.model.model[-1].nc, encoding=encoding
            )
        elif self.mode == SEGMENT_MODE:
            self.make_nn_archive(
                class_list=names,
                n_classes=self.model.model[-1].nc,
                parser="YOLOExtendedParser",
                n_prototypes=self.model.model[-1].nm,
                is_softmax=False,  # E2E outputs are already sigmoided
                output_kwargs={
                    "mask_outputs": ["mask_output"],
                    "protos_outputs": "protos_output",
                },
                encoding=encoding,
            )
        elif self.mode == POSE_MODE:
            self.make_nn_archive(
                class_list=names,
                n_classes=self.model.model[-1].nc,
                parser="YOLOExtendedParser",
                n_keypoints=self.model.model[-1].kpt_shape[0],
                output_kwargs={"keypoints_outputs": ["kpt_output"]},
                encoding=encoding,
            )
