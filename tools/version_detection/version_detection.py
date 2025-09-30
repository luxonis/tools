from __future__ import annotations

import platform
import shutil
import subprocess
from os import listdir
from os.path import exists, isdir, join

YOLOV5_CONVERSION = "yolov5"
YOLOV5U_CONVERSION = "yolov5u"
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
        # Remove and recreate the extracted_model directory
        if exists("extracted_model"):
            shutil.rmtree("extracted_model")
        subprocess.check_output("mkdir extracted_model", shell=True)

        # Extract the tar file into the extracted_model directory
        if platform.system() == "Windows":
            subprocess.check_output(["tar", "-xf", path, "-C", "extracted_model"])
        else:
            subprocess.check_output(["unzip", path, "-d", "extracted_model"])

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
                "yolov8" in content
                or (
                    "YOLOv8" in content and "yolov5" not in content
                )  # the second condition is to avoid yolov5u being detected as yolov8
                or ("v8DetectionLoss" in content and "ultralytics" in content)
                or (
                    "ultralytics.nn.modules.head.Detect"
                    in content  # v8 detection head used not present in v5/v7
                    and "ultralytics.nn.modules.block.C2f"
                    in content  # v8 specific block (C2f = Cross Concat & Fusion)
                    and "ultralytics.nn.modules.block.SPPF"
                    in content  # SPP/SPPF layers appear in v5 too, but this namespace ties it to v8
                )
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
            elif "yolov5u" in content or (
                "yolov5" in content and "ultralytics.nn.modules" in content
                # the second condition checks if the new version of the Ultralytics package was used to build the model which signals the "u" variant
            ):
                return YOLOV5U_CONVERSION
            elif (
                "yolov5" in content
                or "SPPF" in content
                or (
                    "models.yolo.Detectr1" in content
                    and "models.common.SPPr" in content
                )
            ):
                return YOLOV5_CONVERSION

    except subprocess.CalledProcessError as e:
        raise RuntimeError() from e
    finally:
        # Ensure the extracted_model directory is removed after processing
        if exists("extracted_model"):
            shutil.rmtree("extracted_model")

    return UNRECOGNIZED
