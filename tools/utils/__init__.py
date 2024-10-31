from __future__ import annotations

from .config import Config
from .filesystem_utils import (
    download_from_remote,
    get_protocol,
    resolve_path,
    upload_file_to_remote,
)
from .in_channels import get_first_conv2d_in_channels
from .version_detection import (
    GOLD_YOLO_CONVERSION,
    UNRECOGNIZED,
    YOLOV5_CONVERSION,
    YOLOV6R1_CONVERSION,
    YOLOV6R3_CONVERSION,
    YOLOV6R4_CONVERSION,
    YOLOV7_CONVERSION,
    YOLOV8_CONVERSION,
    YOLOV9_CONVERSION,
    YOLOV10_CONVERSION,
    YOLOV11_CONVERSION,
    detect_version,
)

__all__ = [
    "Config",
    "detect_version",
    "YOLOV5_CONVERSION",
    "YOLOV6R1_CONVERSION",
    "YOLOV6R3_CONVERSION",
    "YOLOV6R4_CONVERSION",
    "YOLOV7_CONVERSION",
    "YOLOV8_CONVERSION",
    "YOLOV9_CONVERSION",
    "YOLOV10_CONVERSION",
    "YOLOV11_CONVERSION",
    "GOLD_YOLO_CONVERSION",
    "UNRECOGNIZED",
    "resolve_path",
    "download_from_remote",
    "upload_file_to_remote",
    "get_protocol",
    "get_first_conv2d_in_channels",
]
