from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest
from helper_functions import download_model, nn_archive_checker

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _prepare_model(model_name: str, test_workspace: Path) -> str:
    """Return an absolute model path inside the current worker workspace."""
    weights_dir = test_workspace / "weights"
    model_path = weights_dir / f"{model_name}.pt"

    if not model_path.exists():
        download_model(model_name, str(weights_dir))

    return str(model_path.resolve())


def _run_tools(
    command: list[str],
    test_workspace: Path,
) -> subprocess.CompletedProcess:
    """Run one conversion inside the isolated worker workspace."""
    return subprocess.run(
        command,
        cwd=test_workspace,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _output_dir(test_workspace: Path) -> str:
    return str(test_workspace / "shared_with_container" / "outputs")


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
def test_explicit_version(
    model_info: tuple[str, str],
    test_workspace: Path,
):
    """Tests setting explicit version."""
    model_name = model_info[0]
    version = model_info[1]
    model_path = _prepare_model(model_name, test_workspace)

    command = ["tools", model_path, "--version", version]
    if version == "yolov5":  # edge case when stride=64 is needed
        command += ["--imgsz", "320"]
    logger.debug(f"CLI command: {command}")

    result = _run_tools(command, test_workspace)
    if result.returncode != 0:
        pytest.fail(f"Exit code: {result.returncode}, Output: {result.stdout}")


@pytest.mark.parametrize(
    "version, expected_exit_code",
    [
        ("yolo", 1),
        ("yolov10", 4),
    ],
)
def test_wrong_explicit_version(
    version: str,
    expected_exit_code: int,
    test_workspace: Path,
):
    """Tests setting wrong version - either not valid or not compatible"""
    model_name = "yolov8n"
    model_path = _prepare_model(model_name, test_workspace)

    command = ["tools", model_path, "--version", version]
    logger.debug(f"CLI command: {command}")

    result = _run_tools(command, test_workspace)
    assert result.returncode == expected_exit_code, (
        f"Expected exit code {expected_exit_code}, got {result.returncode}. "
    )


@pytest.mark.parametrize("input_size", ["64", "64 64", "128 64"])
def test_explicit_input_size(
    input_size: str,
    test_workspace: Path,
):
    """Tests setting explicit input size."""
    model_name = "yolov8n"
    model_path = _prepare_model(model_name, test_workspace)

    command = ["tools", model_path, "--imgsz", input_size]
    logger.debug(f"CLI command: {command}")

    result = _run_tools(command, test_workspace)
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
    nn_archive_checker(
        extra_keys_to_check=extra_keys_to_check,
        output_dir=_output_dir(test_workspace),
    )


@pytest.mark.parametrize("input_size", ["64 a", "a", "a a"])
def test_wrong_explicit_input_size(
    input_size: str,
    test_workspace: Path,
):
    """Tests setting wrong explicit input size."""
    model_name = "yolov8n"
    expected_exit_code = 2
    model_path = _prepare_model(model_name, test_workspace)

    command = ["tools", model_path, "--imgsz", input_size]
    logger.debug(f"CLI command: {command}")

    result = _run_tools(command, test_workspace)
    assert result.returncode == expected_exit_code, (
        f"Expected to fail for invalid --imgsz `{input_size}`, got exit code: {result.returncode}. "
    )


@pytest.mark.parametrize("encoding", ["RGB", "BGR", "rgb", "bgr"])
def test_explicit_encoding(
    encoding: str,
    test_workspace: Path,
):
    """Tests setting explicit encoding."""
    model_name = "yolov8n"
    model_path = _prepare_model(model_name, test_workspace)

    command = ["tools", model_path, "--encoding", encoding]
    logger.debug(f"CLI command: {command}")

    result = _run_tools(command, test_workspace)
    if result.returncode != 0:
        pytest.fail(f"Exit code: {result.returncode}, Output: {result.stdout}")

    extra_keys_to_check = [
        (
            ["model", "inputs", 0, "preprocessing", "dai_type"],
            "RGB888p" if encoding.lower() == "rgb" else "BGR888p",
        )
    ]
    nn_archive_checker(
        extra_keys_to_check=extra_keys_to_check,
        output_dir=_output_dir(test_workspace),
    )


@pytest.mark.parametrize("encoding", ["gray", "a"])
def test_wrong_explicit_encoding(
    encoding: str,
    test_workspace: Path,
):
    """Tests setting wrong explicit encoding."""
    model_name = "yolov8n"
    model_path = _prepare_model(model_name, test_workspace)

    command = ["tools", model_path, "--encoding", encoding]
    logger.debug(f"CLI command: {command}")

    result = _run_tools(command, test_workspace)
    assert result.returncode != 0, (
        f"Expected to fail for invalid --encoding `{encoding}`, got exit code: {result.returncode}. "
    )


def test_explicit_class_names(test_workspace: Path):
    """Tests setting explicit class_names."""
    model_name = "yolov8n"
    model_path = _prepare_model(model_name, test_workspace)

    class_names = ["a"] * 80
    class_names_str = ", ".join(class_names)
    command = ["tools", model_path, "--class-names", class_names_str]
    logger.debug(f"CLI command: {command}")

    result = _run_tools(command, test_workspace)
    if result.returncode != 0:
        pytest.fail(f"Exit code: {result.returncode}, Output: {result.stdout}")

    extra_keys_to_check = [
        (
            ["model", "heads", 0, "metadata", "classes"],
            class_names,
        )
    ]
    nn_archive_checker(
        extra_keys_to_check=extra_keys_to_check,
        output_dir=_output_dir(test_workspace),
    )


def test_wrong_explicit_class_names(test_workspace: Path):
    """Tests setting wrong explicit class names."""
    model_name = "yolov8n"
    expected_exit_code = 6
    model_path = _prepare_model(model_name, test_workspace)

    class_names_str = "a"
    command = ["tools", model_path, "--class-names", class_names_str]
    logger.debug(f"CLI command: {command}")

    result = _run_tools(command, test_workspace)
    assert result.returncode == expected_exit_code, (
        f"Expected to fail for invalid --class-names `{class_names_str}`, got exit code: {result.returncode}. "
    )


def test_explicit_version_detection(test_workspace: Path):
    """Tests setting explicit version detection."""
    model_name = "yolov8n"
    model_path = _prepare_model(model_name, test_workspace)

    command = ["tools", model_path]
    logger.debug(f"CLI command: {command}")

    result = _run_tools(command, test_workspace)
    if result.returncode != 0:
        pytest.fail(f"Exit code: {result.returncode}, Output: {result.stdout}")
