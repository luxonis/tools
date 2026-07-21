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


def test_e2e_shard_count_three_manifest_matches_profile():
    import importlib

    e2e_shards = importlib.import_module("e2e_shards")

    assignment = e2e_shards.get_e2e_shard_assignment(3)
    counts = tuple(len(shard) for shard in assignment)
    public_union = set().union(*assignment)
    public_overlap = (
        (assignment[0] & assignment[1])
        | (assignment[0] & assignment[2])
        | (assignment[1] & assignment[2])
    )

    assert e2e_shards.DEFAULT_E2E_SHARD_COUNT == 3
    assert e2e_shards.supported_e2e_shard_counts() == (3, 10)
    assert counts == (33, 35, 33)
    assert len(public_union) == 101
    assert public_overlap == set()


def test_e2e_shard_count_ten_manifest_matches_selected_profile():
    import importlib

    e2e_shards = importlib.import_module("e2e_shards")

    count_three = e2e_shards.get_e2e_shard_assignment(3)
    assignment = e2e_shards.get_e2e_shard_assignment(10)

    counts = tuple(len(shard) for shard in assignment)
    public_union = set().union(*assignment)
    count_three_union = set().union(*count_three)

    seen = set()
    overlap = set()

    for shard in assignment:
        overlap.update(seen & shard)
        seen.update(shard)

    assert len(assignment) == 10
    assert counts == (11, 9, 9, 11, 11, 10, 9, 11, 11, 9)
    assert all(assignment)
    assert len(public_union) == 101
    assert public_union == count_three_union
    assert overlap == set()


def test_e2e_representative_shard_manifest():
    import importlib

    e2e_shards = importlib.import_module("e2e_shards")

    assignment = e2e_shards.get_e2e_shard_assignment(
        2,
        "representative",
    )
    counts = tuple(len(shard) for shard in assignment)
    representative_nodeids = set().union(*assignment)
    full_nodeids = set().union(*e2e_shards.get_e2e_shard_assignment(10))

    assert e2e_shards.supported_e2e_suites() == (
        "full",
        "representative",
    )
    assert e2e_shards.supported_e2e_shard_counts("representative") == (2,)
    assert counts == (14, 16)
    assert len(representative_nodeids) == 30
    assert representative_nodeids < full_nodeids
    assert assignment[0].isdisjoint(assignment[1])


def test_e2e_shard_assignment_detects_collection_drift():
    import importlib

    e2e_shards = importlib.import_module("e2e_shards")
    assignment = e2e_shards.get_e2e_shard_assignment(10)
    public_nodeids = set().union(*assignment)

    with pytest.raises(
        pytest.UsageError,
        match="Missing from assignment",
    ):
        e2e_shards.validate_e2e_shard_assignment(
            public_nodeids | {"tests/test_end2end.py::test_new_public_case"},
            assignment,
        )

    with pytest.raises(
        pytest.UsageError,
        match="Stale in assignment",
    ):
        e2e_shards.validate_e2e_shard_assignment(
            public_nodeids - {next(iter(public_nodeids))},
            assignment,
        )


def test_e2e_shard_count_requires_checked_in_assignment():
    import importlib

    e2e_shards = importlib.import_module("e2e_shards")

    with pytest.raises(pytest.UsageError, match="Regenerate the shard assignment"):
        e2e_shards.get_e2e_shard_assignment(2)
