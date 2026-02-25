from __future__ import annotations

import json
import logging
import os
import tarfile
from pathlib import Path

import requests
from constants import MODEL_TYPE2URL
from luxonis_ml.utils import LuxonisFileSystem

logger = logging.getLogger()


def download_model(model_name: str, folder: str):
    logger.info(f"Downloading model '{model_name}' into folder '{folder}'")
    url = MODEL_TYPE2URL.get(model_name)
    if not url:
        raise ValueError(f"No URL defined for model {model_name}")
    response = requests.get(url)
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{model_name}.pt")
    with open(file_path, "wb") as f:
        f.write(response.content)
    logger.debug(f"Model downloaded and saved to {file_path}")
    return file_path


def download_private_model(model_name: str, filename: str, folder: str) -> str:
    """Download a private model from the GCP bucket using LuxonisFileSystem."""

    logger.info(f"Downloading private model '{model_name}' from GCP bucket")
    remote_path = f"gs://luxonis-test-bucket/tools-tests/{filename}"
    dest_dir = Path(folder)
    dest_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path = LuxonisFileSystem.download(remote_path, dest=dest_dir)
    final_path = dest_dir / f"{model_name}.pt"

    # Cases where name != filename in the PRIVATE_TEST_MODELS entry
    if downloaded_path != final_path and downloaded_path.exists():
        downloaded_path.rename(final_path)

    logger.debug(f"Private model downloaded and saved to {final_path}")
    return str(final_path)


def nn_archive_checker(extra_keys_to_check: list | None = None):
    """Checks only explicitly requested NNArchive config values."""
    assert extra_keys_to_check, "`extra_keys_to_check` must include at least one check."

    config_data = load_latest_nn_archive_config()
    for keys, target in extra_keys_to_check:
        temp_cfg = config_data
        for key in keys[:-1]:
            temp_cfg = temp_cfg[key]
        assert temp_cfg[keys[-1]] == target, (
            f"Value `{temp_cfg[keys[-1]]}` at key `{keys}` doesn't match expected value `{target}`"
        )


def load_latest_nn_archive_config() -> dict:
    """Load config.json from the most recently exported NNArchive."""
    output_dir = "shared_with_container/outputs"
    subdirs = [
        d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))
    ]
    assert subdirs, f"No folders found in `{output_dir}`"

    subdirs.sort(key=lambda d: os.path.getmtime(os.path.join(output_dir, d)))
    latest_subdir = subdirs[-1]
    model_output_path = os.path.join(output_dir, latest_subdir)

    archive_files = [f for f in os.listdir(model_output_path) if f.endswith(".tar.xz")]
    assert len(archive_files) == 1, (
        f"Expected 1 .tar.xz file, found {len(archive_files)}: {archive_files}"
    )
    archive_path = os.path.join(model_output_path, archive_files[0])

    with tarfile.open(archive_path, "r:xz") as tar:
        file_names = [m.name for m in tar.getmembers() if m.isfile()]
        config_files = [name for name in file_names if name.endswith("config.json")]
        assert len(config_files) == 1, (
            f"Expected 1 config.json file, found {len(config_files)}: {config_files}"
        )
        config_member = tar.getmember(config_files[0])
        config_file = tar.extractfile(config_member)
        assert config_file is not None, "Failed to extract config.json"
        return json.load(config_file)
