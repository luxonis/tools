from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from tools.version_detection import (
    GOLD_YOLO_CONVERSION,
    YOLOV5_CONVERSION,
    YOLOV5U_CONVERSION,
    YOLOV6R1_CONVERSION,
    YOLOV6R3_CONVERSION,
    YOLOV6R4_CONVERSION,
    YOLOV7_CONVERSION,
    YOLOV8_CONVERSION,
    YOLOV9_CONVERSION,
    YOLOV10_CONVERSION,
    YOLOV11_CONVERSION,
    YOLOV12_CONVERSION,
    YOLOV26_CONVERSION,
    YOLOV26_NMS_CONVERSION,
    YOLOV26_SEM_CONVERSION,
    YOLOX_CONVERSION,
)

ExporterFactory = Callable[[str, tuple[int, int], bool], Any]


@dataclass(frozen=True)
class ConversionSpec:
    exporter_family: str
    exporter_factory: ExporterFactory


def _build_yolov5_exporter(
    model_path: str, imgsz: tuple[int, int], use_rvc2: bool
) -> Any:
    from tools.yolo.yolov5_exporter import YoloV5Exporter

    return YoloV5Exporter(model_path, imgsz, use_rvc2)


def _build_yolov6r1_exporter(
    model_path: str, imgsz: tuple[int, int], use_rvc2: bool
) -> Any:
    from tools.yolov6r1.yolov6_r1_exporter import YoloV6R1Exporter

    return YoloV6R1Exporter(model_path, imgsz, use_rvc2)


def _build_yolov6r3_exporter(
    model_path: str, imgsz: tuple[int, int], use_rvc2: bool
) -> Any:
    from tools.yolov6r3.yolov6_r3_exporter import YoloV6R3Exporter

    return YoloV6R3Exporter(model_path, imgsz, use_rvc2)


def _build_goldyolo_exporter(
    model_path: str, imgsz: tuple[int, int], use_rvc2: bool
) -> Any:
    from tools.yolov6r3.gold_yolo_exporter import GoldYoloExporter

    return GoldYoloExporter(model_path, imgsz, use_rvc2)


def _build_yolov6r4_exporter(
    model_path: str, imgsz: tuple[int, int], use_rvc2: bool
) -> Any:
    from tools.yolo.yolov6_exporter import YoloV6R4Exporter

    return YoloV6R4Exporter(model_path, imgsz, use_rvc2)


def _build_yolov7_exporter(
    model_path: str, imgsz: tuple[int, int], use_rvc2: bool
) -> Any:
    from tools.yolov7.yolov7_exporter import YoloV7Exporter

    return YoloV7Exporter(model_path, imgsz, use_rvc2)


def _build_yolov8_exporter(
    model_path: str, imgsz: tuple[int, int], use_rvc2: bool
) -> Any:
    from tools.yolo.yolov8_exporter import YoloV8Exporter

    return YoloV8Exporter(model_path, imgsz, use_rvc2)


def _build_yolo26_exporter(
    model_path: str, imgsz: tuple[int, int], use_rvc2: bool
) -> Any:
    from tools.yolo.yolo26_exporter import Yolo26Exporter

    return Yolo26Exporter(model_path, imgsz, use_rvc2)


def _build_yolov10_exporter(
    model_path: str, imgsz: tuple[int, int], use_rvc2: bool
) -> Any:
    from tools.yolo.yolov10_exporter import YoloV10Exporter

    return YoloV10Exporter(model_path, imgsz, use_rvc2)


def _build_yolox_exporter(
    model_path: str, imgsz: tuple[int, int], use_rvc2: bool
) -> Any:
    from tools.yolox.yolox_exporter import YoloXExporter

    return YoloXExporter(model_path, imgsz, use_rvc2)


CONVERSION_SPECS: dict[str, ConversionSpec] = {
    GOLD_YOLO_CONVERSION: ConversionSpec("goldyolo", _build_goldyolo_exporter),
    YOLOV5_CONVERSION: ConversionSpec("yolov5", _build_yolov5_exporter),
    YOLOV5U_CONVERSION: ConversionSpec("yolov8", _build_yolov8_exporter),
    YOLOV6R1_CONVERSION: ConversionSpec("yolov6r1", _build_yolov6r1_exporter),
    YOLOV6R3_CONVERSION: ConversionSpec("yolov6r3", _build_yolov6r3_exporter),
    YOLOV6R4_CONVERSION: ConversionSpec("yolov6r4", _build_yolov6r4_exporter),
    YOLOV7_CONVERSION: ConversionSpec("yolov7", _build_yolov7_exporter),
    YOLOV8_CONVERSION: ConversionSpec("yolov8", _build_yolov8_exporter),
    YOLOV9_CONVERSION: ConversionSpec("yolov8", _build_yolov8_exporter),
    YOLOV10_CONVERSION: ConversionSpec("yolov10", _build_yolov10_exporter),
    YOLOV11_CONVERSION: ConversionSpec("yolov8", _build_yolov8_exporter),
    YOLOV12_CONVERSION: ConversionSpec("yolov8", _build_yolov8_exporter),
    YOLOV26_CONVERSION: ConversionSpec("yolo26", _build_yolo26_exporter),
    YOLOV26_NMS_CONVERSION: ConversionSpec("yolov8", _build_yolov8_exporter),
    YOLOV26_SEM_CONVERSION: ConversionSpec("yolo26", _build_yolo26_exporter),
    YOLOX_CONVERSION: ConversionSpec("yolox", _build_yolox_exporter),
}


def is_supported_version(version: str) -> bool:
    return version in CONVERSION_SPECS


def get_supported_versions() -> tuple[str, ...]:
    return tuple(CONVERSION_SPECS)


def get_exporter_family(version: str) -> str:
    return CONVERSION_SPECS[version].exporter_family


def create_exporter(
    version: str, model_path: str, imgsz: tuple[int, int], use_rvc2: bool
) -> Any:
    return CONVERSION_SPECS[version].exporter_factory(model_path, imgsz, use_rvc2)
