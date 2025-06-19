from __future__ import annotations

import logging
import os

import pytest
import requests
from constants import MODEL_TYPE2URL, SAVE_FOLDER, TEST_MODELS
from typer.testing import CliRunner

from tools.main import app

runner = CliRunner()

# Logging
LOG_FILE = "test_cli_conversion.log"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(LOG_FILE, mode="w")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(console_handler)


@pytest.mark.parametrize(
    "model",
    TEST_MODELS,
    ids=[model["name"] for model in TEST_MODELS],
)
def test_cli_conversion(model: dict, test_config: dict):
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

    command = [model_path]
    logger.debug(f"CLI command: {command}")

    result = runner.invoke(app, command)
    if result.exit_code != 0:
        fail_test_output(result)


@pytest.mark.parametrize(
    "model",
    TEST_MODELS,
    ids=[model["name"] for model in TEST_MODELS],
)
def test_cli_conversion_rvc3(model: dict, test_config: dict):
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

    command = [model_path, "--no-use-rvc2"]
    logger.debug(f"CLI command: {command}")

    result = runner.invoke(app, command)
    if result.exit_code != 0:
        fail_test_output(result)


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

    command = [model_path, "--version", version]
    logger.debug(f"CLI command: {command}")

    result = runner.invoke(app, command)
    if result.exit_code != 0:
        fail_test_output(result)


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

    command = [model_path, "--version", version]
    logger.debug(f"CLI command: {command}")

    result = runner.invoke(app, command)
    assert result.exit_code == expected_exit_code, (
        f"Expected exit code {expected_exit_code}, got {result.exit_code}. "
    )


@pytest.mark.parametrize("input_size", ["64", "64 64", "128 64"])
def test_explicit_input_size(input_size: str):
    """Tests setting explicit input size"""
    model_name = "yolov8n"
    model_path = os.path.join(SAVE_FOLDER, f"{model_name}.pt")
    if not os.path.exists(model_path):
        download_model(model_name, SAVE_FOLDER)

    command = [model_path, "--imgsz", input_size]
    logger.debug(f"CLI command: {command}")

    result = runner.invoke(app, command)
    if result.exit_code != 0:
        fail_test_output(result)


@pytest.mark.parametrize("input_size", ["64 a", "a", "a a"])
def test_wrong_explicit_input_size(input_size: str):
    """Tests setting wrong explicit input size"""
    model_name = "yolov8n"
    expected_exit_code = 2
    model_path = os.path.join(SAVE_FOLDER, f"{model_name}.pt")
    if not os.path.exists(model_path):
        download_model(model_name, SAVE_FOLDER)

    command = [model_path, "--imgsz", input_size]
    logger.debug(f"CLI command: {command}")

    result = runner.invoke(app, command)
    assert result.exit_code == expected_exit_code, (
        f"Expected exit code {expected_exit_code}, got {result.exit_code}. "
    )


@pytest.mark.parametrize("encoding", ["RGB", "BGR"])
def test_explicit_encoding(encoding: str):
    """Tests setting explicit encoding"""
    model_name = "yolov8n"
    model_path = os.path.join(SAVE_FOLDER, f"{model_name}.pt")
    if not os.path.exists(model_path):
        download_model(model_name, SAVE_FOLDER)

    command = [model_path, "--encoding", encoding]
    logger.debug(f"CLI command: {command}")

    result = runner.invoke(app, command)
    if result.exit_code != 0:
        fail_test_output(result)


@pytest.mark.parametrize("encoding", ["rgb", "gray", "a"])
def test_wrong_explicit_encoding(encoding: str):
    """Tests setting wrong explicit encoding"""
    model_name = "yolov8n"
    model_path = os.path.join(SAVE_FOLDER, f"{model_name}.pt")
    if not os.path.exists(model_path):
        download_model(model_name, SAVE_FOLDER)

    command = [model_path, "--encoding", encoding]
    logger.debug(f"CLI command: {command}")

    result = runner.invoke(app, command)
    assert result.exit_code != 0, (
        f"Expected to fail for invalid --encoding value `{encoding}`"
    )


def test_explicit_class_names():
    """Tests setting explicit class_names"""
    model_name = "yolov8n"
    model_path = os.path.join(SAVE_FOLDER, f"{model_name}.pt")
    if not os.path.exists(model_path):
        download_model(model_name, SAVE_FOLDER)

    class_names = ["a"] * 80
    class_names_str = ", ".join(class_names)
    command = [model_path, "--class-names", class_names_str]
    logger.debug(f"CLI command: {command}")

    result = runner.invoke(app, command)
    if result.exit_code != 0:
        fail_test_output(result)


def test_wrong_explicit_class_names():
    """Tests setting wrong explicit class names"""
    model_name = "yolov8n"
    expected_exit_code = 6
    model_path = os.path.join(SAVE_FOLDER, f"{model_name}.pt")
    if not os.path.exists(model_path):
        download_model(model_name, SAVE_FOLDER)

    class_names_str = "a"
    command = [model_path, "--class-names", class_names_str]
    logger.debug(f"CLI command: {command}")

    result = runner.invoke(app, command)
    assert result.exit_code == expected_exit_code, (
        f"Expected to fail for invalid --class-names value `{class_names_str}`"
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


def fail_test_output(cli_result):
    last_line = (
        cli_result.output.strip().splitlines()[-1]
        if cli_result.output.strip()
        else "<no output>"
    )
    pytest.fail(f"Exit code: {cli_result.exit_code}, Output: {last_line}")
