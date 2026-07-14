from __future__ import annotations

import os
import pathlib
from pathlib import Path

from luxonis_ml.utils import LuxonisFileSystem

from tools.utils.constants import SHARED_DIR


def patch_pathlib_for_cross_platform():
    """Patch ``pathlib`` path classes for cross-platform pickle loading.

    This allows objects pickled on Windows to be loaded on POSIX systems and vice versa
    by aliasing the platform-specific path implementation.
    """
    if os.name == "nt":  # Windows
        pathlib.PosixPath = pathlib.WindowsPath
    else:  # Linux macOS and WSL
        pathlib.WindowsPath = pathlib.PosixPath


def resolve_path(string: str, dest: Path) -> Path:
    """Resolve a local or remote model path to an existing local file.

    Remote paths are downloaded into ``dest``. Local paths are used directly.
    If the path is relative and does not exist, ``SHARED_DIR`` is used as a
    fallback root.

    Args:
        string: Local path or remote URI.
        dest: Directory used for remote downloads.

    Returns:
        A local path that exists on disk.

    Raises:
        ValueError: If the resolved path does not exist.
    """
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


def download_from_remote(url: str, dest: Path | str, max_files: int = -1) -> Path:
    """Download a file or directory from remote bucket storage.

    If ``url`` points to a directory, files are downloaded under ``dest`` while
    preserving the remote subpath. When ``max_files`` is set, only that many
    files are fetched from the directory walk.

    Args:
        url: Remote filesystem URL.
        dest: Local destination directory.
        max_files: Maximum number of files to download from a directory. Use
            ``-1`` to download all files.

    Returns:
        The local path to the downloaded file or directory root.
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
    local_path: Path | str, url: str, put_file_plugin: str | None = None
) -> None:
    """Upload a local file to remote bucket storage.

    Args:
        local_path: Path to the local file.
        url: Remote destination URL.
        put_file_plugin: Optional filesystem plugin override used for upload.
    """

    absolute_path, remote_path = LuxonisFileSystem.split_full_path(url)
    fs = LuxonisFileSystem(absolute_path, put_file_plugin=put_file_plugin)

    fs.put_file(str(local_path), remote_path)


def get_protocol(url: str) -> str:
    """Return the filesystem protocol for a URL.

    Args:
        url: Local path or remote filesystem URL.

    Returns:
        The resolved filesystem protocol.
    """

    return LuxonisFileSystem.get_protocol(url)
