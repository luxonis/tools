# ruff: noqa: I001
from __future__ import annotations

import os
import sys
from collections.abc import Mapping

import torch
import torch.nn as nn

from tools.modules import Exporter
from tools.utils.constants import Encoding

current_dir = os.path.dirname(os.path.abspath(__file__))
yolox_path = os.path.join(current_dir, "YOLOX")
if yolox_path not in sys.path:
    sys.path.insert(0, yolox_path)

from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead  # noqa: E402
from yolox.models.network_blocks import SiLU  # noqa: E402
from yolox.utils import replace_module  # noqa: E402


COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class DetectYOLOX(nn.Module):
    """Expose YOLOX's three raw detection maps for archive postprocessing."""

    def __init__(self, head: YOLOXHead):
        super().__init__()
        self.nc = head.num_classes
        self.nl = len(head.strides)
        self.stride = head.strides
        self.stems = head.stems
        self.cls_convs = head.cls_convs
        self.reg_convs = head.reg_convs
        self.cls_preds = head.cls_preds
        self.reg_preds = head.reg_preds
        self.obj_preds = head.obj_preds

    def forward(self, features: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        outputs = []
        for index in range(self.nl):
            x = self.stems[index](features[index])
            cls_output = self.cls_preds[index](self.cls_convs[index](x))
            reg_features = self.reg_convs[index](x)
            reg_output = self.reg_preds[index](reg_features)
            obj_output = self.obj_preds[index](reg_features)
            outputs.append(
                torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], dim=1
                )
            )
        return tuple(outputs)


def _infer_standard_architecture(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[float, float, bool]:
    """Infer a standard YOLOX architecture from its head width."""
    try:
        head_channels = state_dict["head.cls_preds.0.weight"].shape[1]
    except KeyError as exc:
        raise ValueError("Checkpoint does not contain a YOLOX detection head.") from exc

    # Maps detection-head channel count to (depth multiplier, width multiplier, depthwise).
    architectures = {
        64: (0.33, 0.25, True),  # YOLOX-Nano
        96: (0.33, 0.375, False),  # YOLOX-Tiny
        128: (0.33, 0.50, False),  # YOLOX-S
        192: (0.67, 0.75, False),  # YOLOX-M
        256: (1.00, 1.00, False),  # YOLOX-L
        320: (1.33, 1.25, False),  # YOLOX-X
    }
    try:
        return architectures[head_channels]
    except KeyError as exc:
        raise ValueError(
            "Unsupported YOLOX head width. Only standard YOLOX-Nano/Tiny/S/M/L/X "
            "checkpoints are currently supported."
        ) from exc


class YoloXExporter(Exporter):
    """Export standard YOLOX detection checkpoints as Luxonis NNArchives."""

    strides = [8, 16, 32]
    output_names = ["output1_yolov6", "output2_yolov6", "output3_yolov6"]

    def __init__(
        self,
        model_path: str,
        imgsz: tuple[int, int],
        use_rvc2: bool,
    ):
        super().__init__(
            model_path,
            imgsz,
            use_rvc2,
            # YOLOX uses the same grid decode as this already-supported subtype.
            subtype="yolov6r1",
            output_names=self.output_names,
        )
        self.load_model()

    def load_model(self) -> None:
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model", checkpoint)
        if not isinstance(state_dict, Mapping):
            raise ValueError("YOLOX checkpoint must contain a model state dictionary.")

        depth, width, depthwise = _infer_standard_architecture(state_dict)
        try:
            num_classes = state_dict["head.cls_preds.0.weight"].shape[0]
        except KeyError as exc:
            raise ValueError(
                "Checkpoint does not contain YOLOX class predictions."
            ) from exc

        backbone = YOLOPAFPN(depth=depth, width=width, act="silu", depthwise=depthwise)
        head = YOLOXHead(
            num_classes=num_classes, width=width, act="silu", depthwise=depthwise
        )
        model = YOLOX(backbone, head)
        model.load_state_dict(state_dict, strict=True)
        model = replace_module(model, nn.SiLU, SiLU)
        model.head = DetectYOLOX(model.head)
        model.eval()

        if any(size % max(self.strides) != 0 for size in self.imgsz):
            raise ValueError("Image size must be divisible by the maximum stride (32).")

        self.number_of_channels = 3
        self.nc = num_classes
        self.names = (
            COCO_CLASSES
            if num_classes == len(COCO_CLASSES)
            else [f"Class_{index}" for index in range(num_classes)]
        )
        self.model = model

    def export_nn_archive(
        self, class_names: list[str] | None = None, encoding: Encoding = Encoding.BGR
    ) -> None:
        """Create an NNArchive with the existing YOLOv6 R1 grid decoder."""
        names = self.names
        if class_names is not None:
            assert len(class_names) == self.nc, (
                f"Number of given class names {len(class_names)} does not match "
                f"the model class count {self.nc}."
            )
            names = class_names

        self.make_nn_archive(
            class_list=names,
            n_classes=self.nc,
            parser="YOLOExtendedParser",
            output_kwargs={"strides": self.strides},
            encoding=encoding,
            mean=[0, 0, 0],
            scale=[1, 1, 1],
        )
