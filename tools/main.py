from __future__ import annotations

import logging
from typing import Optional, List

import typer

from tools.utils import (
    GOLD_YOLO_CONVERSION,
    YOLOV5_CONVERSION,
    YOLOV6R1_CONVERSION,
    YOLOV6R3_CONVERSION,
    YOLOV6R4_CONVERSION,
    YOLOV7_CONVERSION,
    YOLOV8_CONVERSION,
    YOLOV10_CONVERSION,
    Config,
    detect_version,
    upload_file_to_remote,
    resolve_path,
)
from tools.utils.constants import MISC_DIR


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


app = typer.Typer(help="Tools CLI", add_completion=False, rich_markup_mode="markdown")


YOLO_VERSIONS = [
    GOLD_YOLO_CONVERSION,
    YOLOV5_CONVERSION,
    YOLOV6R1_CONVERSION,
    YOLOV6R3_CONVERSION,
    YOLOV6R4_CONVERSION,
    YOLOV7_CONVERSION,
    YOLOV8_CONVERSION,
    YOLOV10_CONVERSION,
]


@app.command()
def convert(
    model: str,
    imgsz: str = "416 416",
    version: Optional[str] = None,
    use_rvc2: bool = True,
    class_names: Optional[str] = None,
    output_remote_url: Optional[str] = None,
    config_path: Optional[str] = None,
    put_file_plugin: Optional[str] = None,
):
    logger = logging.getLogger(__name__)
    logger.info("Converting model...")

    if version is not None and version not in YOLO_VERSIONS:
        logger.error("Wrong YOLO version selected!")
        raise typer.Exit(code=1)

    try:
        imgsz = list(map(int, imgsz.split(" "))) if " " in imgsz else [int(imgsz)] * 2
    except ValueError:
        logger.error('Invalid image size format. Must be "width height" or "width".')
        raise typer.Exit(code=1)

    if class_names:
        class_names = [class_name.strip() for class_name in class_names.split(",")]
        logger.info(f"Class names: {class_names}")

    config = Config.get_config(
        config_path,
        {
            "model": model,
            "imgsz": imgsz,
            "use_rvc2": use_rvc2,
            "class_names": class_names,
            "output_remote_url": output_remote_url,
            "put_file_plugin": put_file_plugin,
        },
    )

    # Resolve model path
    model_path = resolve_path(config.model, MISC_DIR)

    if version is None:
        version = detect_version(str(model_path))
        logger.info(f"Detected version: {version}")

    try:
        # Create exporter
        logger.info("Loading model...")
        if version == YOLOV5_CONVERSION:
            from tools.yolo.yolov5_exporter import YoloV5Exporter
            exporter = YoloV5Exporter(str(model_path), config.imgsz, config.use_rvc2)
        elif version == YOLOV6R1_CONVERSION:
            from tools.yolov6r1.yolov6_r1_exporter import YoloV6R1Exporter
            exporter = YoloV6R1Exporter(str(model_path), config.imgsz, config.use_rvc2)
        elif version == YOLOV6R3_CONVERSION:
            from tools.yolov6r3.yolov6_r3_exporter import YoloV6R3Exporter
            exporter = YoloV6R3Exporter(str(model_path), config.imgsz, config.use_rvc2)
        elif version == GOLD_YOLO_CONVERSION:
            from tools.yolov6r3.gold_yolo_exporter import GoldYoloExporter
            exporter = GoldYoloExporter(str(model_path), config.imgsz, config.use_rvc2)
        elif version == YOLOV6R4_CONVERSION:
            from tools.yolo.yolov6_exporter import YoloV6R4Exporter
            exporter = YoloV6R4Exporter(str(model_path), config.imgsz, config.use_rvc2)
        elif version == YOLOV7_CONVERSION:
            from tools.yolov7.yolov7_exporter import YoloV7Exporter
            exporter = YoloV7Exporter(str(model_path), config.imgsz, config.use_rvc2)
        elif version == YOLOV8_CONVERSION:
            from tools.yolo.yolov8_exporter import YoloV8Exporter
            exporter = YoloV8Exporter(str(model_path), config.imgsz, config.use_rvc2)
        elif version == YOLOV10_CONVERSION:
            from tools.yolo.yolov10_exporter import YoloV10Exporter
            exporter = YoloV10Exporter(str(model_path), config.imgsz, config.use_rvc2)
        else:
            logger.error("Unrecognized model version.")
            raise typer.Exit(code=1)
        logger.info("Model loaded.")
    except Exception as e:
        logger.error(f"Error creating exporter: {e}")
        raise typer.Exit(code=1)
    
    # Export model
    try:
        logger.info("Exporting model...")
        exporter.export_onnx()
        logger.info("Model exported.")
    except Exception as e:
        logger.error(f"Error exporting model: {e}")
        raise typer.Exit(code=1)
    # Create NN archive
    try:
        logger.info("Creating NN archive...")
        exporter.export_nn_archive(config.class_names)
        logger.info("NN archive created.")
    except Exception as e:
        logger.error(f"Error creating NN archive: {e}")
        raise typer.Exit(code=1)
    
    # Upload to remote
    if config.output_remote_url:
        upload_file_to_remote(exporter.f_nn_archive, config.output_remote_url, config.put_file_plugin)
        logger.info(f"Uploaded NN archive to {config.output_remote_url}")


if __name__ == "__main__":
    app()
