from __future__ import annotations

from .config import Config
from .version_detection import (
    GOLD_YOLO_CONVERSION,
    UNRECOGNIZED,
    YOLOV5_CONVERSION,
    YOLOV6R1_CONVERSION,
    YOLOV6R3_CONVERSION,
    YOLOV6R4_CONVERSION,
    YOLOV7_CONVERSION,
    YOLOV8_CONVERSION,
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
    "GOLD_YOLO_CONVERSION",
    "UNRECOGNIZED",
]
