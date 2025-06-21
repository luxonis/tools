from __future__ import annotations

import logging
import os
import subprocess

import pytest
from constants import SAVE_FOLDER
from helper_functions import download_model, nn_archive_checker

logger = logging.getLogger()
logger.setLevel(logging.INFO)

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
    """Tests setting explicit version."""
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
    """Tests setting explicit input size."""
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
    """Tests setting wrong explicit input size."""
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
    """Tests setting explicit encoding."""
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
    """Tests setting wrong explicit encoding."""
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
    """Tests setting explicit class_names."""
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
    """Tests setting wrong explicit class names."""
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
