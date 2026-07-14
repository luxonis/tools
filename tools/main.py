#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional, cast

from cyclopts import App, Parameter
from loguru import logger
from luxonis_ml.utils import setup_logging
from typing_extensions import Annotated

from tools.utils import (
    Config,
    resolve_path,
    upload_file_to_remote,
)
from tools.utils.constants import MISC_DIR, Encoding
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
    detect_version,
)

setup_logging()

app = App(help="Tools CLI", help_format="markdown", version_flags=())


YOLO_VERSIONS = [
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
]


@app.command()
def convert(
    model: Annotated[str, Parameter()],
    /,
    *,
    imgsz: Annotated[
        str,
        Parameter(show_default=True),
    ] = "416 416",
    version: Annotated[
        Optional[str],
        Parameter(show_default=True),
    ] = None,
    encoding: Annotated[
        Encoding,
        Parameter(show_default=True),
    ] = Encoding.RGB,
    use_rvc2: Annotated[
        bool,
        Parameter(show_default=True),
    ] = True,
    class_names: Annotated[
        Optional[str],
        Parameter(show_default=True),
    ] = None,
    output_remote_url: Annotated[
        Optional[str],
        Parameter(show_default=True),
    ] = None,
    put_file_plugin: Annotated[
        Optional[str],
        Parameter(show_default=True),
    ] = None,
):
    """Convert a supported YOLO model into Luxonis NNArchive.

    The command resolves the input model path, detects the model family when
    necessary, exports ONNX, builds an NN archive, and can optionally upload the
    resulting artifact to remote storage.

    Args:
        model: Path or remote URI to the model file to convert.
        imgsz: Input image size as either ``"width height"`` or a single
            ``"size"`` value applied to both dimensions.
        version: YOLO variant to force, such as ``"yolov8"``. When omitted, the
            command runs automatic version detection.
        encoding: Color encoding used by the input model. Must be ``RGB`` or
            ``BGR``.
        use_rvc2: Whether to target RVC2 instead of RVC3.
        class_names: Comma-separated class names recognized by the model.
        output_remote_url: Remote destination URL for uploading the generated NN
            archive.
        put_file_plugin: Name of a function registered in
            ``PUT_FILE_REGISTRY`` for uploads.

    Raises:
        SystemExit: Exits with a non-zero status when validation, exporter
            creation, export, or archive generation fails.
    """
    if version is not None and version not in YOLO_VERSIONS:
        logger.error("Wrong YOLO version selected!")
        raise SystemExit(1) from None

    try:
        imgsz_list = (
            list(map(int, imgsz.split(" "))) if " " in imgsz else [int(imgsz)] * 2
        )
    except ValueError as e:
        logger.error('Invalid image size format. Must be "width height" or "size".')
        raise SystemExit(2) from e

    if class_names:
        class_names_list = [class_name.strip() for class_name in class_names.split(",")]
        logger.info(f"Class names: {class_names_list}")
    else:
        class_names_list = class_names

    config = Config.get_config(
        {
            "model": model,
            "imgsz": imgsz_list,
            "encoding": encoding,
            "use_rvc2": use_rvc2,
            "class_names": class_names_list,
            "output_remote_url": output_remote_url,
            "put_file_plugin": put_file_plugin,
        }
    )
    exporter_imgsz = cast(tuple[int, int], tuple(config.imgsz))

    # Resolve model path
    model_path = resolve_path(config.model, MISC_DIR)
    if version is None:
        version = detect_version(str(model_path))
        version_note = (
            "(This is an anchor-free version of the YOLOv5 model obtained by a more recent version of Ultralytics. Therefore, YOLOv8 conversion will be used instead of the standard YOLOv5 conversion)"
            if version == YOLOV5U_CONVERSION
            else ""
        )
        logger.info(f"Detected version: {version} {version_note}")

    try:
        # Create exporter
        logger.info("Loading model...")
        if version == YOLOV5_CONVERSION:
            from tools.yolo.yolov5_exporter import YoloV5Exporter

            exporter = YoloV5Exporter(str(model_path), exporter_imgsz, config.use_rvc2)
        elif version == YOLOV6R1_CONVERSION:
            from tools.yolov6r1.yolov6_r1_exporter import YoloV6R1Exporter

            exporter = YoloV6R1Exporter(
                str(model_path), exporter_imgsz, config.use_rvc2
            )
        elif version == YOLOV6R3_CONVERSION:
            from tools.yolov6r3.yolov6_r3_exporter import YoloV6R3Exporter

            exporter = YoloV6R3Exporter(
                str(model_path), exporter_imgsz, config.use_rvc2
            )
        elif version == GOLD_YOLO_CONVERSION:
            from tools.yolov6r3.gold_yolo_exporter import GoldYoloExporter

            exporter = GoldYoloExporter(
                str(model_path), exporter_imgsz, config.use_rvc2
            )
        elif version == YOLOV6R4_CONVERSION:
            from tools.yolo.yolov6_exporter import YoloV6R4Exporter

            exporter = YoloV6R4Exporter(
                str(model_path), exporter_imgsz, config.use_rvc2
            )
        elif version == YOLOV7_CONVERSION:
            from tools.yolov7.yolov7_exporter import YoloV7Exporter

            exporter = YoloV7Exporter(str(model_path), exporter_imgsz, config.use_rvc2)
        elif version in [
            YOLOV5U_CONVERSION,
            YOLOV8_CONVERSION,
            YOLOV9_CONVERSION,
            YOLOV11_CONVERSION,
            YOLOV12_CONVERSION,
            YOLOV26_NMS_CONVERSION,
        ]:
            from tools.yolo.yolov8_exporter import YoloV8Exporter

            exporter = YoloV8Exporter(str(model_path), exporter_imgsz, config.use_rvc2)
        elif version in [YOLOV26_CONVERSION, YOLOV26_SEM_CONVERSION]:
            from tools.yolo.yolo26_exporter import Yolo26Exporter

            exporter = Yolo26Exporter(str(model_path), exporter_imgsz, config.use_rvc2)
        elif version == YOLOV10_CONVERSION:
            from tools.yolo.yolov10_exporter import YoloV10Exporter

            exporter = YoloV10Exporter(str(model_path), exporter_imgsz, config.use_rvc2)
        else:
            logger.error("Unrecognized model version.")
            raise SystemExit(3) from None
        logger.info("Model loaded.")
    except Exception as e:
        logger.error(f"Error creating exporter: {e}")
        raise SystemExit(4) from e

    # Export model
    try:
        logger.info("Exporting model...")
        exporter.export_onnx()
        logger.info("Model exported.")
    except Exception as e:
        logger.error(f"Error exporting model: {e}")
        raise SystemExit(5) from e
    # Create NN archive
    try:
        logger.info("Creating NN archive...")
        exporter.export_nn_archive(
            class_names=config.class_names, encoding=config.encoding
        )
        logger.info(f"NN archive created in {exporter.output_folder}.")
    except Exception as e:
        logger.error(f"Error creating NN archive: {e}")
        raise SystemExit(6) from e

    # Upload to remote
    if config.output_remote_url:
        archive_path = exporter.f_nn_archive
        if archive_path is None:
            raise RuntimeError("NN archive path is missing after archive generation.")
        upload_file_to_remote(
            archive_path, config.output_remote_url, config.put_file_plugin
        )
        logger.info(f"Uploaded NN archive to {config.output_remote_url}")


if __name__ == "__main__":
    app()
