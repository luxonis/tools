from __future__ import annotations

import importlib
import os
import sys
import tarfile
import zipfile
from contextlib import contextmanager
from os import listdir
from os.path import isdir, join
from pathlib import Path
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
GOLD_YOLO_CONVERSION = "goldyolo"
UNRECOGNIZED = "none"

CURRENT_DIR = Path(__file__).resolve().parent.parent


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

    except (tarfile.TarError, zipfile.BadZipFile, ValueError, OSError) as e:
        raise RuntimeError(f"Failed to extract archive: {e}") from e
    finally:
        # Ensure the extracted_model directory is removed after processing
        temp_dir.cleanup()

    return detect_by_loading(path)  # Fallback to model loading detection


@contextmanager
def yolo_repo_import(repo_root: Path, cleanup: bool = True):
    """Temporarily add a YOLO repo to sys.path and isolate its imports.

    Any modules imported while inside this context will be removed from
    sys.modules afterwards, so different repos with the same names
    (e.g. `models.experimental`) don't clash.
    """
    repo_root = Path(repo_root).resolve()

    if cleanup:
        old_modules = set(sys.modules.keys())

    sys.path.insert(0, str(repo_root))

    try:
        yield
    finally:
        sys.path.remove(str(repo_root))

        if cleanup:
            new_modules = set(sys.modules.keys()) - old_modules
            repo_root_str = str(repo_root)
            for name in new_modules:
                mod = sys.modules.get(name)
                filename = getattr(mod, "__file__", None)
                if filename and repo_root_str in os.path.abspath(filename):
                    sys.modules.pop(name, None)


def fix_yolov6_imports():
    for name in list(sys.modules):
        if name == "yolov6" or name.startswith("yolov6."):
            del sys.modules[name]

    module = importlib.import_module("yolov6.utils.checkpoint")
    return module.load_checkpoint


def fix_yolov5_yolov7_imports():
    for name in list(sys.modules):
        if name.startswith("models.") or name.startswith("utils."):
            del sys.modules[name]

    module = importlib.import_module("models.experimental")
    return module.attempt_load


def try_yolov5_load(weights_path: str):
    with yolo_repo_import(CURRENT_DIR / "yolo" / "yolov5"):
        fix_yolov5_yolov7_imports()
        from tools.yolo.yolov5_exporter import YoloV5Exporter

        exporter = YoloV5Exporter(weights_path, [320, 320], True)
        return exporter


def try_yolov6r1_load(weights_path: str):
    with yolo_repo_import(CURRENT_DIR / "yolov6r1" / "YOLOv6R1", cleanup=False):
        fix_yolov6_imports()
        from tools.yolov6r1.yolov6_r1_exporter import YoloV6R1Exporter

        exporter = YoloV6R1Exporter(weights_path, [320, 320], True)
        return exporter


def try_yolov6r3_load(weights_path: str):
    with yolo_repo_import(CURRENT_DIR / "yolov6r3" / "YOLOv6R3", cleanup=False):
        fix_yolov6_imports()
        from tools.yolov6r3.yolov6_r3_exporter import YoloV6R3Exporter

        exporter = YoloV6R3Exporter(weights_path, [320, 320], True)
        return exporter


def try_yolov6r4_load(weights_path: str):
    with yolo_repo_import(CURRENT_DIR / "yolo" / "YOLOv6", cleanup=False):
        fix_yolov6_imports()
        from tools.yolo.yolov6_exporter import YoloV6R4Exporter

        exporter = YoloV6R4Exporter(weights_path, [320, 320], True)
        return exporter


def try_yolov7_load(weights_path: str):
    with yolo_repo_import(CURRENT_DIR / "yolov7" / "yolov7"):
        fix_yolov5_yolov7_imports()
        from tools.yolov7.yolov7_exporter import YoloV7Exporter

        exporter = YoloV7Exporter(weights_path, [320, 320], True)
        return exporter


def try_goldyolo_load(weights_path: str):
    with yolo_repo_import(CURRENT_DIR / "yolov6r3" / "Efficient-Computing"):
        from tools.yolov6r3.gold_yolo_exporter import GoldYoloExporter

        exporter = GoldYoloExporter(weights_path, [320, 320], True)
        return exporter


def detect_by_loading(path: str) -> str:
    """Try to detect YOLO version by attempting to load the model with different
    loaders.

    Returns the version name if successful, otherwise returns UNRECOGNIZED.
    """

    candidates = [
        (YOLOV5_CONVERSION, try_yolov5_load),
        (YOLOV6R1_CONVERSION, try_yolov6r1_load),
        (YOLOV6R3_CONVERSION, try_yolov6r3_load),
        (YOLOV6R4_CONVERSION, try_yolov6r4_load),
        (YOLOV7_CONVERSION, try_yolov7_load),
        (GOLD_YOLO_CONVERSION, try_goldyolo_load),
    ]

    for version_name, loader in candidates:
        try:
            loader(path)
            return version_name
        except Exception:
            continue

    return UNRECOGNIZED
