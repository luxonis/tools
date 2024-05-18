from pathlib import Path
from typing import Final

SHARED_DIR: Final[Path] = Path("shared_with_container")
OUTPUTS_DIR: Final[Path] = SHARED_DIR / "outputs"
MISC_DIR: Final[Path] = SHARED_DIR / "misc"

__all__ = ["SHARED_DIR", "OUTPUTS_DIR", "MISC_DIR"]
