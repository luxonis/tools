from __future__ import annotations

import copy
import json
import logging
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import requests
from constants import MODEL_TYPE2URL
from luxonis_ml.utils import LuxonisFileSystem

logger = logging.getLogger()


def get_shared_dir() -> Path:
    return Path(os.environ.get("TOOLS_SHARED_DIR", "shared_with_container"))


def get_outputs_dir() -> Path:
    return get_shared_dir() / "outputs"


def get_tools_command(*args: str) -> list[str]:
    return [sys.executable, "-m", "tools.main", *args]


def download_model(model_name: str, folder: str):
    logger.info(f"Downloading model '{model_name}' into folder '{folder}'")
    url = MODEL_TYPE2URL.get(model_name)
    if not url:
        raise ValueError(f"No URL defined for model {model_name}")
    response = requests.get(url)
    os.makedirs(folder, exist_ok=True)
    file_path = Path(folder) / f"{model_name}.pt"
    with tempfile.NamedTemporaryFile(
        dir=folder, prefix=f"{model_name}.", suffix=".tmp", delete=False
    ) as f:
        f.write(response.content)
        temp_path = Path(f.name)
    os.replace(temp_path, file_path)
    logger.debug(f"Model downloaded and saved to {file_path}")
    return str(file_path)


def download_private_model(model_name: str, filename: str, folder: str) -> str:
    """Download a private model from the GCP bucket using LuxonisFileSystem."""

    logger.info(f"Downloading private model '{model_name}' from GCP bucket")
    remote_path = f"gs://luxonis-test-bucket/tools-tests/{filename}"
    dest_dir = Path(folder)
    dest_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = dest_dir / f".tmp-{model_name}-{os.getpid()}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = LuxonisFileSystem.download(remote_path, dest=temp_dir)
    final_path = dest_dir / f"{model_name}.pt"

    # Cases where name != filename in the PRIVATE_TEST_MODELS entry
    if downloaded_path != final_path and downloaded_path.exists():
        os.replace(downloaded_path, final_path)

    if temp_dir.exists():
        temp_dir.rmdir()

    logger.debug(f"Private model downloaded and saved to {final_path}")
    return str(final_path)


def nn_archive_checker(extra_keys_to_check: list = []):  # noqa: B006
    """Tests the content of the exported NNArchive."""
    output_dir = get_outputs_dir()
    assert output_dir.exists(), f"Output directory `{output_dir}` does not exist"
    subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
    assert subdirs, f"No folders found in `{output_dir}`"

    # Sort by modification time (most recent last)
    subdirs.sort(key=lambda d: d.stat().st_mtime)
    latest_subdir = subdirs[-1]
    model_output_path = latest_subdir
    logger.debug(f"Model output path: {model_output_path}")

    # Find .tar.xz archive
    archive_files = [
        f.name for f in model_output_path.iterdir() if f.name.endswith(".tar.xz")
    ]
    assert len(archive_files) == 1, (
        f"Expected 1 .tar.xz file, found {len(archive_files)}: {archive_files}"
    )
    archive_path = model_output_path / archive_files[0]

    # Open and inspect the archive
    with tarfile.open(archive_path, "r:xz") as tar:
        members = tar.getmembers()
        file_names = [m.name for m in members if m.isfile()]

        # Find .onnx and config.json
        onnx_files = [name for name in file_names if name.endswith(".onnx")]
        config_files = [name for name in file_names if name.endswith("config.json")]

        assert len(onnx_files) == 1, (
            f"Expected 1 .onnx file, found {len(onnx_files)}: {onnx_files}"
        )
        assert len(config_files) == 1, (
            f"Expected 1 config.json file, found {len(config_files)}: {config_files}"
        )

        # Load and check config.json
        config_member = tar.getmember(config_files[0])
        config_file = tar.extractfile(config_member)
        assert config_file is not None, "Failed to extract config.json"

        config_data = json.load(config_file)
        # Check content of the NNArchive
        assert config_data["model"]["metadata"]["path"] == onnx_files[0], (
            f"Path in config.json `{config_data['model']['metadata']['path']}` doesn't match the ONNX path `{onnx_files[0]}`"
        )
        expected_mean = [0.0, 0.0, 0.0]
        assert (
            config_data["model"]["inputs"][0]["preprocessing"]["mean"] == expected_mean
        ), (
            f"Inputs mean `{config_data['model']['inputs'][0]['preprocessing']['mean']}` doesn't match the expected mean `{expected_mean}`"
        )
        expected_scale = [255.0, 255.0, 255.0]
        assert (
            config_data["model"]["inputs"][0]["preprocessing"]["scale"]
            == expected_scale
        ), (
            f"Inputs scale `{config_data['model']['inputs'][0]['preprocessing']['scale']}` doesn't match the expected scale `{expected_scale}`"
        )

        if len(extra_keys_to_check) and not any(
            ["dai_type" in i for i in extra_keys_to_check[0]]
        ):  # only check if we are not already checking though "extra_keys_to_check"
            dai_type = "RGB888p"
            assert (
                config_data["model"]["inputs"][0]["preprocessing"]["dai_type"]
                == dai_type
            ), (
                f"Inputs dai_type `{config_data['model']['inputs'][0]['preprocessing']['dai_type']}` doesn't match the expected dai_type `{dai_type}`"
            )

        if extra_keys_to_check:
            temp_cfg = copy.deepcopy(config_data)
            for keys, target in extra_keys_to_check:
                for key in keys[:-1]:
                    temp_cfg = temp_cfg[key] if isinstance(key, str) else temp_cfg[key]
                assert temp_cfg[keys[-1]] == target, (
                    f"Value `{temp_cfg[keys[-1]]}` at key `{keys}` doesn't match expected value `{target}`"
                )


def load_latest_nn_archive_config() -> dict:
    """Load config.json from the most recently exported NNArchive."""
    output_dir = get_outputs_dir()
    assert output_dir.exists(), f"Output directory `{output_dir}` does not exist"
    subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
    assert subdirs, f"No folders found in `{output_dir}`"

    subdirs.sort(key=lambda d: d.stat().st_mtime)
    latest_subdir = subdirs[-1]
    model_output_path = latest_subdir

    archive_files = [
        f.name for f in model_output_path.iterdir() if f.name.endswith(".tar.xz")
    ]
    assert len(archive_files) == 1, (
        f"Expected 1 .tar.xz file, found {len(archive_files)}: {archive_files}"
    )
    archive_path = model_output_path / archive_files[0]

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
