from __future__ import annotations

from typing import Literal

from luxonis_ml.utils import LuxonisConfig
from pydantic import Field, field_validator

from tools.utils.constants import Encoding


class Config(LuxonisConfig):
    """Configuration for a model conversion run."""

    model: str = Field(..., description="Path to the model's weights")
    imgsz: list[int] = Field(
        default=[416, 416],
        min_length=2,
        max_length=2,
        description="Input image size [width, height].",
    )
    encoding: Encoding = Field(
        default=Encoding.RGB, description="Color encoding used in the input model."
    )
    class_names: list[str] | None = Field(None, description="List of class names.")
    use_rvc2: Literal[False, True] = Field(True, description="Whether to use RVC2.")
    output_remote_url: str | None = Field(
        None, description="URL to upload the output to."
    )
    put_file_plugin: str | None = Field(
        None,
        description="The name of a registered function under the PUT_FILE_REGISTRY.",
    )

    @field_validator("imgsz", mode="before")
    @classmethod
    def check_imgsz(cls, value):
        """Validate that the image size is positive and stride-compatible.

        Args:
            value: Candidate width and height values.

        Returns:
            The validated image size values.

        Raises:
            ValueError: If any dimension is non-positive or not divisible by 32.
        """
        if any([v <= 0 for v in value]):
            raise ValueError("Image size values must be greater than 0.")
        if any([v % 32 != 0 for v in value]):
            raise ValueError("Image size values must be divisible by 32.")
        return value
