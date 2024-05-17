from __future__ import annotations

import logging
from typing import Optional

import typer
from typing_extensions import Annotated, TypeAlias

from tools.utils import (
    GOLD_YOLO_CONVERSION,
    YOLOV5_CONVERSION,
    YOLOV6R1_CONVERSION,
    YOLOV6R3_CONVERSION,
    YOLOV6R4_CONVERSION,
    YOLOV7_CONVERSION,
    YOLOV8_CONVERSION,
    Config,
    detect_version,
)

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


app = typer.Typer(help="Tools CLI", add_completion=False, rich_markup_mode="markdown")


ImgszOption: TypeAlias = Annotated[
    str, typer.Option(help='Image size "width height" or "width".')
]
ModelPathOption: TypeAlias = Annotated[
    str, typer.Option(help="Path to the model's weights.")
]
UseRVC2Option: TypeAlias = Annotated[bool, typer.Option(help="Whether to use RVC2.")]
OutputRemoteUrlOption: TypeAlias = Annotated[
    Optional[str], typer.Option(help="URL to upload the output to.")
]
ConfigPathOption: TypeAlias = Annotated[
    Optional[str], typer.Option(help="Path to the config file.")
]


@app.command()
def convert(
    model: ModelPathOption,
    imgsz: ImgszOption = "416 416",
    use_rvc2: UseRVC2Option = True,
    output_remote_url: OutputRemoteUrlOption = None,
    config_path: ConfigPathOption = None,
):
    logger = logging.getLogger(__name__)
    logger.info("Converting model...")

    try:
        imgsz = list(map(int, imgsz.split(" "))) if " " in imgsz else [int(imgsz)] * 2
    except ValueError:
        logger.error('Invalid image size format. Must be "width height" or "width".')
        raise typer.Exit(code=1)

    config = Config.get_config(
        config_path,
        {
            "model": model,
            "imgsz": imgsz,
            "use_rvc2": use_rvc2,
            "output_remote_url": output_remote_url,
        },
    )
    logger.info(f"Config: {config}")

    version = detect_version(config.model)
    logger.info(f"Detected version: {version}")

    if version == YOLOV5_CONVERSION:
        pass
    elif version == YOLOV6R1_CONVERSION:
        pass
    elif version == YOLOV6R3_CONVERSION:
        pass
    elif version == GOLD_YOLO_CONVERSION:
        pass
    elif version == YOLOV6R4_CONVERSION:
        pass
    elif version == YOLOV7_CONVERSION:
        pass
    elif version == YOLOV8_CONVERSION:
        pass
    else:
        logger.error("Unrecognized model version.")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

# python main.py --model ../yolov8n-seg.pt --imgsz "416"
# tools --model yolov8n-seg.pt --imgsz "416"
