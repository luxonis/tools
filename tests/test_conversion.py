from __future__ import annotations

import os

import requests

from tools.utils.version_detection import detect_version
from tools.yolo.yolov6_exporter import YoloV6R4Exporter
from tools.yolo.yolov8_exporter import YoloV8Exporter
from tools.yolo.yolov10_exporter import YoloV10Exporter
from tools.yolov7.yolov7_exporter import YoloV7Exporter


def _download_file(url: str):
    """An util function for downloading file from the given URL and saving it in the current folder."""
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Determine the filename from the URL or use the provided new_filename
        filename = url.split("/")[-1]

        # Construct the full path to save the file
        file_path = os.path.join("tests", filename)

        # Write the content of the response to the file
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"File downloaded and saved as {file_path}")
    else:
        print("Failed to download the file")


def _remove_file(file_path: str):
    """An util function for removing a file from the current folder."""
    if os.path.exists(file_path):
        os.remove(file_path)


def _test_model_conversion(exported_class, model_path, imgsz, use_rvc2):
    """Test the conversion of a model."""
    exporter = exported_class(model_path, imgsz, use_rvc2)
    exporter.export_onnx()
    exporter.export_nn_archive()

    # Check that the output files exist and are not empty
    assert os.path.exists(str(exporter.f_onnx))
    assert os.path.exists(str(exporter.f_nn_archive))
    _remove_file(model_path)


def test_yolov5n_automatic_version_detection():
    """Test the YOLOv5n autodetection of the model version."""
    _download_file(
        "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt"
    )
    assert detect_version("tests/yolov5n.pt") == "yolov5"
    _remove_file("tests/yolov5n.pt")


def test_yolov5nu_automatic_version_detection():
    """Test the YOLOv5nu autodetection of the model version."""
    _download_file(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5nu.pt"
    )
    assert detect_version("tests/yolov5nu.pt") == "yolov5u"
    _remove_file("tests/yolov5nu.pt")


def test_yolov5nu_model_conversion():
    """Test the conversion of an YOLOv5nu model."""
    _download_file(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5nu.pt"
    )
    _test_model_conversion(YoloV8Exporter, "tests/yolov5nu.pt", (64, 64), True)


def test_yolov6nr1_automatic_version_detection():
    """Test the YOLOv6nr1 autodetection of the model version."""
    _download_file(
        "https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6n.pt"
    )
    assert detect_version("tests/yolov6n.pt") == "yolov6r1"
    _remove_file("tests/yolov6n.pt")


def test_yolov6nr2_automatic_version_detection():
    """Test the YOLOv6nr2 autodetection of the model version."""
    _download_file(
        "https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6n.pt"
    )
    assert detect_version("tests/yolov6n.pt") == "yolov6r3"
    _remove_file("tests/yolov6n.pt")


def test_yolov6nr3_automatic_version_detection():
    """Test the YOLOv6nr3 autodetection of the model version."""
    _download_file(
        "https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6n.pt"
    )
    assert detect_version("tests/yolov6n.pt") == "yolov6r3"
    _remove_file("tests/yolov6n.pt")


def test_yolov6nr4_automatic_version_detection():
    """Test the YOLOv6nr4 autodetection of the model version."""
    _download_file(
        "https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6n.pt"
    )
    assert detect_version("tests/yolov6n.pt") == "yolov6r4"
    _remove_file("tests/yolov6n.pt")


def test_yolov6nr4_model_conversion():
    """Test the conversion of an YOLOv6nr4 model."""
    _download_file(
        "https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6n.pt"
    )
    _test_model_conversion(YoloV6R4Exporter, "tests/yolov6n.pt", (640, 480), True)


def test_yolov7t_automatic_version_detection():
    """Test the YOLOv7t autodetection of the model version."""
    _download_file(
        "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt"
    )
    assert detect_version("tests/yolov7-tiny.pt") == "yolov7"
    _remove_file("tests/yolov7-tiny.pt")


def test_yolov7t_model_conversion():
    """Test the conversion of an YOLOv7t model."""
    _download_file(
        "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt"
    )
    _test_model_conversion(YoloV7Exporter, "tests/yolov7-tiny.pt", (640, 480), True)


def test_yolov8n_automatic_version_detection():
    """Test the YOLOv8n autodetection of the model version."""
    _download_file(
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    )
    assert detect_version("tests/yolov8n.pt") == "yolov8"
    _remove_file("tests/yolov8n.pt")


def test_yolov8n_model_conversion():
    """Test the conversion of an YOLOv8n model."""
    _download_file(
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    )
    _test_model_conversion(YoloV8Exporter, "tests/yolov8n.pt", (640, 480), True)


def test_yolov9t_automatic_version_detection():
    """Test the YOLOv9t autodetection of the model version."""
    _download_file(
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9t.pt"
    )
    assert detect_version("tests/yolov9t.pt") == "yolov9"
    _remove_file("tests/yolov9t.pt")


def test_yolov9t_model_conversion():
    """Test the conversion of an YOLOv9t model."""
    _download_file(
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9t.pt"
    )
    _test_model_conversion(YoloV8Exporter, "tests/yolov9t.pt", (640, 480), True)


def test_yolov10n_automatic_version_detection():
    """Test the YOLOv10n autodetection of the model version."""
    _download_file(
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt"
    )
    assert detect_version("tests/yolov10n.pt") == "yolov10"
    _remove_file("tests/yolov10n.pt")


def test_yolov10n_model_conversion():
    """Test the conversion of an YOLOv10n model."""
    _download_file(
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt"
    )
    _test_model_conversion(YoloV10Exporter, "tests/yolov10n.pt", (640, 480), True)


def test_yolov11n_automatic_version_detection():
    """Test the YOLOv11n autodetection of the model version."""
    _download_file(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
    )
    assert detect_version("tests/yolo11n.pt") == "yolov11"
    _remove_file("tests/yolo11n.pt")


def test_yolov11n_model_conversion():
    """Test the conversion of an YOLOv11n model."""
    _download_file(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
    )
    _test_model_conversion(YoloV8Exporter, "tests/yolo11n.pt", (640, 480), True)


def test_yolov11n_cls_automatic_version_detection():
    """Test the YOLOv11n cls autodetection of the model version."""
    _download_file(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt"
    )
    assert detect_version("tests/yolo11n-cls.pt") == "yolov11"
    _remove_file("tests/yolo11n-cls.pt")


def test_yolov11n_cls_model_conversion():
    """Test the conversion of an YOLOv11n cls model."""
    _download_file(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt"
    )
    _test_model_conversion(YoloV8Exporter, "tests/yolo11n-cls.pt", (224, 224), True)


def test_yolov11n_seg_automatic_version_detection():
    """Test the YOLOv11n seg autodetection of the model version."""
    _download_file(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt"
    )
    assert detect_version("tests/yolo11n-seg.pt") == "yolov11"
    _remove_file("tests/yolo11n-seg.pt")


def test_yolov11n_seg_model_conversion():
    """Test the conversion of an YOLOv11n seg model."""
    _download_file(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt"
    )
    _test_model_conversion(YoloV8Exporter, "tests/yolo11n-seg.pt", (640, 480), True)


def test_yolov11n_obb_automatic_version_detection():
    """Test the YOLOv11n obb autodetection of the model version."""
    _download_file(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.pt"
    )
    assert detect_version("tests/yolo11n-obb.pt") == "yolov11"
    _remove_file("tests/yolo11n-obb.pt")


def test_yolov11n_obb_model_conversion():
    """Test the conversion of an YOLOv11n obb model."""
    _download_file(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.pt"
    )
    _test_model_conversion(YoloV8Exporter, "tests/yolo11n-obb.pt", (640, 480), True)


def test_yolov11n_kpts_automatic_version_detection():
    """Test the YOLOv11n kpts autodetection of the model version."""
    _download_file(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"
    )
    assert detect_version("tests/yolo11n-pose.pt") == "yolov11"
    _remove_file("tests/yolo11n-pose.pt")


def test_yolov11n_kpts_model_conversion():
    """Test the conversion of an YOLOv11n kpts model."""
    _download_file(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"
    )
    _test_model_conversion(YoloV8Exporter, "tests/yolo11n-pose.pt", (640, 480), True)
