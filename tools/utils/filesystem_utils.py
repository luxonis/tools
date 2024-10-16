from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from luxonis_ml.utils import LuxonisFileSystem

from tools.utils.constants import SHARED_DIR


def resolve_path(string: str, dest: Path) -> Path:
    """Downloads the file from remote or returns the path otherwise."""
    protocol = get_protocol(string)
    if protocol != "file":
        path = download_from_remote(string, dest)
    else:
        path = Path(string)
    if not path.exists():
        path = SHARED_DIR / path
    if not path.exists():
        raise ValueError(f"Path `{string}` does not exist.")
    return path


def download_from_remote(url: str, dest: Union[Path, str], max_files: int = -1) -> Path:
    """Downloads file(s) from remote bucket storage.

    It could be single file, entire direcory, or `max_files` within a directory
    """

    absolute_path, remote_path = LuxonisFileSystem.split_full_path(url)
    if isinstance(dest, str):
        dest = Path(dest)
    local_path = dest / remote_path
    fs = LuxonisFileSystem(absolute_path)

    if fs.is_directory(remote_path):
        for i, remote_file in enumerate(fs.walk_dir(remote_path)):
            if i == max_files:
                break
            if not local_path.exists():
                fs.get_file(remote_file, str(local_path / Path(remote_file).name))

    else:
        if not local_path.exists():
            fs.get_file(remote_path, str(local_path))

    return local_path


def upload_file_to_remote(
    local_path: Union[Path, str], url: str, put_file_plugin: Optional[str] = None
) -> None:
    """Uploads a file to remote bucket storage."""

    absolute_path, remote_path = LuxonisFileSystem.split_full_path(url)
    fs = LuxonisFileSystem(absolute_path, put_file_plugin=put_file_plugin)

    fs.put_file(str(local_path), remote_path)


def get_protocol(url: str) -> str:
    """Returns LuxonisFileSystem protocol."""

    return LuxonisFileSystem.get_protocol(url)
