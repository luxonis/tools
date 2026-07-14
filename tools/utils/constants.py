from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Final

SHARED_DIR: Final[Path] = Path("shared_with_container")
OUTPUTS_DIR: Final[Path] = SHARED_DIR / "outputs"
MISC_DIR: Final[Path] = SHARED_DIR / "misc"


class Encoding(str, Enum):
    """Supported color encodings for model inputs."""

    RGB = "RGB"
    BGR = "BGR"

    def get_dai_type(self):
        """Return the corresponding DepthAI image type string.

        Returns:
            The DepthAI image type for the encoding.
        """
        if self == Encoding.RGB:
            return "RGB888p"
        elif self == Encoding.BGR:
            return "BGR888p"


__all__ = ["SHARED_DIR", "OUTPUTS_DIR", "MISC_DIR", "Encoding"]
