from __future__ import annotations

import os

from tools.utils.version_detection import detect_version
from tools.yolo.yolov8_exporter import YoloV8Exporter


def test_automatic_version_detection():
    """Test the autodetection of the model version."""
    assert detect_version("tests/yolov8n.pt") == "yolov8"


def test_model_conversion():
    """Test the conversion of a model."""
    exporter = YoloV8Exporter("tests/yolov8n.pt", (416, 416), True)
    exporter.export_onnx()
    exporter.export_nn_archive()

    # Check that the output files exist
    assert os.path.exists(str(exporter.f_onnx))
    assert os.path.exists(str(exporter.f_nn_archive))
