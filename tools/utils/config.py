from __future__ import annotations

from typing import List, Literal, Optional

from luxonis_ml.utils import LuxonisConfig
from pydantic import Field, field_validator

from tools.utils.constants import Encoding


class Config(LuxonisConfig):
    model: str = Field(..., description="Path to the model's weights")
    imgsz: List[int] = Field(
        default=[416, 416],
        min_length=2,
        max_length=2,
        description="Input image size [width, height].",
    )
    encoding: Encoding = Field(
        default=Encoding.RGB, description="Color encoding used in the input model."
    )
    class_names: Optional[List[str]] = Field(None, description="List of class names.")
    use_rvc2: Literal[False, True] = Field(True, description="Whether to use RVC2.")
    output_remote_url: Optional[str] = Field(
        None, description="URL to upload the output to."
    )
    put_file_plugin: Optional[str] = Field(
        None,
        description="The name of a registered function under the PUT_FILE_REGISTRY.",
    )

    @field_validator("imgsz", mode="before")
    @classmethod
    def check_imgsz(cls, value):
        if any([v <= 0 for v in value]):
            raise ValueError("Image size values must be greater than 0.")
        if any([v % 32 != 0 for v in value]):
            raise ValueError("Image size values must be divisible by 32.")
        return value
