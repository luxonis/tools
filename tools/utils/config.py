from __future__ import annotations

from typing import List, Literal, Optional

from luxonis_ml.utils import LuxonisConfig
from pydantic import Field, validator


class Config(LuxonisConfig):
    model: str = Field(..., description="Path to the model's weights")
    imgsz: List[int] = Field(
        default=[416, 416],
        min_length=2,
        max_length=2,
        min_ledescription="Image size [width, height].",
    )
    use_rvc2: Literal[False, True] = Field(True, description="Whether to use RVC2.")
    output_remote_url: Optional[str] = Field(
        None, description="URL to upload the output to."
    )
    put_file_plugin: Optional[str] = Field(
        None, description="The name of a registered function under the PUT_FILE_REGISTRY."
    )

    @validator("imgsz", each_item=True)
    def check_imgsz(cls, v):
        if v <= 0:
            raise ValueError("Image size values must be greater than 0.")
        if v % 32 != 0:
            raise ValueError("Image size values must be divisible by 32.")
        return v
