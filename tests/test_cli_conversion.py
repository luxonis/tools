from __future__ import annotations

import copy
import json
import logging
import os
import subprocess
import tarfile

import pytest
import requests
from constants import MODEL_TYPE2URL, SAVE_FOLDER, TEST_MODELS

# --- Logger setup ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)


@pytest.mark.parametrize(
    "model",
    TEST_MODELS,
    ids=[model["name"] for model in TEST_MODELS],
)
def test_cli_conversion(model: dict, test_config: dict, subtests):
    """Tests the whole CLI conversion flow with no extra params specified"""
    logger.info(f"Testing model: {model['name']}")

    if (
        test_config["test_case"] is not None
        and model["name"] != test_config["test_case"]
    ):
        pytest.skip(
            f"Test case ({model['name']}) doesn't match selected test case ({test_config['test_case']})"
        )

    if (
        test_config["yolo_version"] is not None
        and model["version"] != test_config["yolo_version"]
    ):
        pytest.skip(
            f"Model version ({model['version']}) doesn't match selected version ({test_config['yolo_version']})."
        )

    download_weights = test_config["download_weights"]

    model_path = os.path.join(SAVE_FOLDER, f"{model['name']}.pt")
    if not os.path.exists(model_path):
        if download_weights:
            model_path = download_model(
                model["name"],
                SAVE_FOLDER,
            )
        else:
            pytest.skip("Weights not present and `download_weights` not set")

    command = ["tools", model_path]
    if model.get("size"):  # edge case when stride=64 is needed
        command += ["--imgsz", model.get("size")]

    # include RVC2 and RVC3 tests
    for i in range(2):
        with subtests.test(msg=f"Testing case: {'use_rvc2' if i == 0 else 'use_rvc3'}"):
            if i == 1:
                command.append("--no-use-rvc2")
            logger.debug(f"CLI command: {command}")

            result = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            if result.returncode != 0:
                pytest.fail(f"Exit code: {result.returncode}, Output: {result.stdout}")

            extra_keys_to_check = (
                [
                    (
                        ["model", "inputs", 0, "shape"],
                        [1, 3, int(model.get("size")), int(model.get("size"))],  # type: ignore
                    )
                ]
                if model.get("size")
                else []
            )
            nn_archive_checker(extra_keys_to_check=extra_keys_to_check)


MODEL_EXPLICIT_VERSION = [
    ("yolov5n", "yolov5"),
    ("yolov5nu", "yolov5u"),
    ("yolov6nr1", "yolov6r1"),
    ("yolov6nr2", "yolov6r3"),
    ("yolov6nr4", "yolov6r4"),
    ("yolov7t", "yolov7"),
    ("yolov8n", "yolov8"),
    ("yolov9s", "yolov9"),
    ("yolov10n", "yolov10"),
    ("yolov11n", "yolov11"),
]


@pytest.mark.parametrize(
    "model_info",
    MODEL_EXPLICIT_VERSION,
    ids=[model[0] for model in MODEL_EXPLICIT_VERSION],
)
def test_explicit_version(model_info: tuple[str, str]):
    """Tests setting explicit version"""
    model_name = model_info[0]
    version = model_info[1]
    model_path = os.path.join(SAVE_FOLDER, f"{model_name}.pt")
    if not os.path.exists(model_path):
        download_model(model_name, SAVE_FOLDER)

    command = ["tools", model_path, "--version", version]
    if version == "yolov5":  # edge case when stride=64 is needed
        command += ["--imgsz", "320"]
    logger.debug(f"CLI command: {command}")

    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if result.returncode != 0:
        pytest.fail(f"Exit code: {result.returncode}, Output: {result.stdout}")


@pytest.mark.parametrize(
    "version, expected_exit_code",
    [
        ("yolo", 1),
        ("yolov10", 4),
    ],
)
def test_wrong_explicit_version(version: str, expected_exit_code: int):
    """Tests setting wrong version - either not valid or not compatible"""
    model_name = "yolov8n"
    model_path = os.path.join(SAVE_FOLDER, f"{model_name}.pt")
    if not os.path.exists(model_path):
        download_model(model_name, SAVE_FOLDER)

    command = ["tools", model_path, "--version", version]
    logger.debug(f"CLI command: {command}")

    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert result.returncode == expected_exit_code, (
        f"Expected exit code {expected_exit_code}, got {result.returncode}. "
    )


@pytest.mark.parametrize("input_size", ["64", "64 64", "128 64"])
def test_explicit_input_size(input_size: str):
    """Tests setting explicit input size"""
    model_name = "yolov8n"
    model_path = os.path.join(SAVE_FOLDER, f"{model_name}.pt")
    if not os.path.exists(model_path):
        download_model(model_name, SAVE_FOLDER)

    command = ["tools", model_path, "--imgsz", input_size]
    logger.debug(f"CLI command: {command}")

    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if result.returncode != 0:
        pytest.fail(f"Exit code: {result.returncode}, Output: {result.stdout}")

    imgsz = (
        list(map(int, input_size.split(" ")))
        if " " in input_size
        else [int(input_size)] * 2
    )
    extra_keys_to_check = [
        (["model", "inputs", 0, "shape"], [1, 3, imgsz[1], imgsz[0]])
    ]
    nn_archive_checker(extra_keys_to_check=extra_keys_to_check)


@pytest.mark.parametrize("input_size", ["64 a", "a", "a a"])
def test_wrong_explicit_input_size(input_size: str):
    """Tests setting wrong explicit input size"""
    model_name = "yolov8n"
    expected_exit_code = 2
    model_path = os.path.join(SAVE_FOLDER, f"{model_name}.pt")
    if not os.path.exists(model_path):
        download_model(model_name, SAVE_FOLDER)

    command = ["tools", model_path, "--imgsz", input_size]
    logger.debug(f"CLI command: {command}")

    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert result.returncode == expected_exit_code, (
        f"Expected to fail for invalid --imgsz `{input_size}`, got exit code: {result.returncode}. "
    )


@pytest.mark.parametrize("encoding", ["RGB", "BGR"])
def test_explicit_encoding(encoding: str):
    """Tests setting explicit encoding"""
    model_name = "yolov8n"
    model_path = os.path.join(SAVE_FOLDER, f"{model_name}.pt")
    if not os.path.exists(model_path):
        download_model(model_name, SAVE_FOLDER)

    command = ["tools", model_path, "--encoding", encoding]
    logger.debug(f"CLI command: {command}")

    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if result.returncode != 0:
        pytest.fail(f"Exit code: {result.returncode}, Output: {result.stdout}")

    extra_keys_to_check = [
        (
            ["model", "inputs", 0, "preprocessing", "dai_type"],
            "RGB888p" if encoding == "RGB" else "BGR888p",
        )
    ]
    nn_archive_checker(extra_keys_to_check=extra_keys_to_check)


@pytest.mark.parametrize("encoding", ["rgb", "gray", "a"])
def test_wrong_explicit_encoding(encoding: str):
    """Tests setting wrong explicit encoding"""
    model_name = "yolov8n"
    model_path = os.path.join(SAVE_FOLDER, f"{model_name}.pt")
    if not os.path.exists(model_path):
        download_model(model_name, SAVE_FOLDER)

    command = ["tools", model_path, "--encoding", encoding]
    logger.debug(f"CLI command: {command}")

    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert result.returncode != 0, (
        f"Expected to fail for invalid --encoding `{encoding}`, got exit code: {result.returncode}. "
    )


def test_explicit_class_names():
    """Tests setting explicit class_names"""
    model_name = "yolov8n"
    model_path = os.path.join(SAVE_FOLDER, f"{model_name}.pt")
    if not os.path.exists(model_path):
        download_model(model_name, SAVE_FOLDER)

    class_names = ["a"] * 80
    class_names_str = ", ".join(class_names)
    command = ["tools", model_path, "--class-names", class_names_str]
    logger.debug(f"CLI command: {command}")

    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if result.returncode != 0:
        pytest.fail(f"Exit code: {result.returncode}, Output: {result.stdout}")

    extra_keys_to_check = [
        (
            ["model", "heads", 0, "metadata", "classes"],
            class_names,
        )
    ]
    nn_archive_checker(extra_keys_to_check=extra_keys_to_check)


def test_wrong_explicit_class_names():
    """Tests setting wrong explicit class names"""
    model_name = "yolov8n"
    expected_exit_code = 6
    model_path = os.path.join(SAVE_FOLDER, f"{model_name}.pt")
    if not os.path.exists(model_path):
        download_model(model_name, SAVE_FOLDER)

    class_names_str = "a"
    command = ["tools", model_path, "--class-names", class_names_str]
    logger.debug(f"CLI command: {command}")

    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert result.returncode == expected_exit_code, (
        f"Expected to fail for invalid --class-names `{class_names_str}`, got exit code: {result.returncode}. "
    )


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


def nn_archive_checker(extra_keys_to_check: list = []):
    """Tests the content of the exported NNArchive"""
    output_dir = "shared_with_container/outputs"
    subdirs = [
        d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))
    ]
    assert subdirs, f"No folders found in `{output_dir}`"

    # Sort by modification time (most recent last)
    subdirs.sort(key=lambda d: os.path.getmtime(os.path.join(output_dir, d)))
    latest_subdir = subdirs[-1]
    model_output_path = os.path.join(output_dir, latest_subdir)
    logger.debug(f"Model output path: {model_output_path}")

    # Find .tar.xz archive
    archive_files = [f for f in os.listdir(model_output_path) if f.endswith(".tar.xz")]
    assert len(archive_files) == 1, (
        f"Expected 1 .tar.xz file, found {len(archive_files)}: {archive_files}"
    )
    archive_path = os.path.join(model_output_path, archive_files[0])

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
