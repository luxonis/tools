from __future__ import annotations

import tarfile
import zipfile
from os import listdir
from os.path import isdir, join
from tempfile import TemporaryDirectory

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
YOLOV12_CONVERSION = "yolov12"
YOLOV26_CONVERSION = "yolov26"
GOLD_YOLO_CONVERSION = "goldyolo"
UNRECOGNIZED = "none"


def _extract_archive(archive_path: str, extract_to: str) -> None:
    """Extract an archive to a specified directory.

    Supports both tar and zip formats, automatically detecting the format.

    Args:
        archive_path (str): Path to the archive file
        extract_to (str): Directory to extract to
    """
    # Try tar first
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(path=extract_to)
        return

    # Try zip
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zip_file:
            zip_file.extractall(path=extract_to)
        return

    # If neither worked, raise an error
    raise ValueError(f"Unsupported archive format: {archive_path}")


def detect_version(path: str, debug: bool = False) -> str:
    """Detect the version of the model weights.

    Args:
        path (str): Path to the model weights

    Returns:
        str: The detected version
    """
    # Create a temporary directory
    temp_dir = TemporaryDirectory()
    temp_dir_path = temp_dir.name

    try:
        # Try to extract the archive using appropriate method
        _extract_archive(path, temp_dir_path)

        folder = [f for f in listdir(temp_dir_path) if isdir(join(temp_dir_path, f))][0]

        if "yolov8" in folder.lower():
            return YOLOV8_CONVERSION

        # open pickled data
        with open(f"{temp_dir_path}/{folder}/data.pkl", "rb") as file:
            data = file.read()
            if debug:
                print(data.decode(errors="replace"))
            content = data.decode("latin1")
            if "yolo26" in content:
                return YOLOV26_CONVERSION
            elif "yolov12" in content:
                return YOLOV12_CONVERSION
            elif "yolo11" in content:
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
            elif (
                "models.yolo" in content
                and "models.common.SP" in content
                and not ("SPPF" in content or "SPPr" in content)
            ):
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

    except (tarfile.TarError, zipfile.BadZipFile, ValueError, OSError) as e:
        raise RuntimeError(f"Failed to extract archive: {e}") from e
    finally:
        # Ensure the extracted_model directory is removed after processing
        temp_dir.cleanup()

    return UNRECOGNIZED
