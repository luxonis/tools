from __future__ import annotations

import subprocess
from os import listdir
from os.path import exists, isdir, join

YOLOV5_CONVERSION = "yolov5"
YOLOV6R1_CONVERSION = "yolov6r1"
YOLOV6R3_CONVERSION = "yolov6r3"
YOLOV6R4_CONVERSION = "yolov6r4"
YOLOV7_CONVERSION = "yolov7"
YOLOV8_CONVERSION = "yolov8"
YOLOV9_CONVERSION = "yolov9"
YOLOV10_CONVERSION = "yolov10"
YOLOV11_CONVERSION = "yolov11"
GOLD_YOLO_CONVERSION = "goldyolo"
UNRECOGNIZED = "none"


def detect_version(path: str, debug: bool = False) -> str:
    """Detect the version of the model weights.

    Args:
        path (str): Path to the model weights

    Returns:
        str: The detected version
    """
    try:
        if exists("extracted_model") and isdir("extracted_model"):
            subprocess.check_output(
                f"rm -r extracted_model && unzip {path} -d extracted_model", shell=True
            )
        else:
            subprocess.check_output(f"unzip {path} -d extracted_model", shell=True)
        folder = [
            f for f in listdir("extracted_model") if isdir(join("extracted_model", f))
        ][0]

        if "yolov8" in folder.lower():
            return YOLOV8_CONVERSION

        # open a file, where you stored the pickled data
        with open(f"extracted_model/{folder}/data.pkl", "rb") as file:
            data = file.read()
            if debug:
                print(data.decode(errors="replace"))
            content = data.decode("latin1")

            if "yolo11" in content:
                return YOLOV11_CONVERSION
            elif "yolov10" in content or "v10DetectLoss" in content:
                return YOLOV10_CONVERSION
            elif "yolov9" in content or (
                "v9-model" in content and "ultralytics" in content
            ):
                return YOLOV9_CONVERSION
            elif (
                "YOLOv5u" in content
                or "YOLOv8" in content
                or "yolov8" in content
                or ("v8DetectionLoss" in content and "ultralytics" in content)
            ):
                return YOLOV8_CONVERSION
            elif "yolov6" in content:
                if "yolov6.models.yolo\nDetect" in content:
                    return YOLOV6R1_CONVERSION
                elif "CSPSPPFModule" in content or "ConvBNReLU" in content:
                    return YOLOV6R4_CONVERSION
                elif "gold_yolo" in content:
                    return GOLD_YOLO_CONVERSION
                return YOLOV6R3_CONVERSION
            elif "yolov7" in content:
                return YOLOV7_CONVERSION
            elif (
                "SPPF" in content
                or "yolov5" in content
                or (
                    "models.yolo.Detectr1" in content
                    and "models.common.SPPr" in content
                )
            ):
                return YOLOV5_CONVERSION

        # Remove the output folder
        subprocess.check_output("rm -r extracted_model", shell=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError() from e

    return UNRECOGNIZED
