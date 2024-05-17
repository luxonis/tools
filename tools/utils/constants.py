from pathlib import Path
from typing import Final

SHARED_DIR: Final[Path] = Path("shared_with_container")
OUTPUTS_DIR: Final[Path] = SHARED_DIR / "outputs"

__all__ = ["SHARED_DIR", "OUTPUTS_DIR"]
