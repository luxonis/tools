import logging
import os
import shutil
from pathlib import Path

import pytest

logger = logging.getLogger()


def pytest_addoption(parser):
    parser.addoption(
        "--download-weights",
        action="store_true",
        help="Download weights if not present",
    )
    parser.addoption(
        "--no-delete-output",
        action="store_true",
        help="Don't delete output files after test",
    )
    parser.addoption(
        "--yolo-version",
        choices=[
            "v5",
            "v6",
            "v6r2",
            "v6r4",
            "v7",
            "v8",
            "v9",
            "v10",
            "v11",
            "v12",
            "v26",
            "v26_nms",
        ],
        default=None,
        help="If set then test only that specific yolo version",
    )
    parser.addoption(
        "--test-case",
        type=str,
        default=None,
        help="If set then test only that specific test case",
    )
    parser.addoption(
        "--delete-weights-now",
        action="store_true",
        help="Clean weights after every test to save space - but longer test time.",
    )
    parser.addoption(
        "--test-private",
        action="store_true",
        help="Run tests for private models hosted in GCP bucket (requires GCP authentication).",
    )


@pytest.fixture(scope="session")
def test_config(pytestconfig):
    return {
        "download_weights": pytestconfig.getoption("download_weights"),
        "delete_output": not pytestconfig.getoption("no_delete_output"),
        "yolo_version": pytestconfig.getoption("yolo_version"),
        "test_case": pytestconfig.getoption("test_case"),
        "delete_weights_now": pytestconfig.getoption("delete_weights_now"),
        "test_private": pytestconfig.getoption("test_private"),
        "weights_dir": Path(__file__).resolve().parents[1] / "weights",
    }


@pytest.fixture(scope="function", autouse=True)
def isolate_test_workspace(monkeypatch, tmp_path):
    shared_dir = tmp_path / "shared_with_container"
    monkeypatch.setenv("TOOLS_SHARED_DIR", str(shared_dir))
    return shared_dir


@pytest.fixture(scope="function", autouse=True)
def cleanup_output_after_tests(test_config, isolate_test_workspace):
    yield  # Tests run here
    if test_config["delete_output"]:
        if isolate_test_workspace.exists():
            shutil.rmtree(isolate_test_workspace)
            logger.info(f"Removed test artifacts from {isolate_test_workspace}")


@pytest.fixture(scope="function", autouse=True)
def cleanup_weights_after_tests(test_config):
    yield  # Tests run here
    if test_config["delete_weights_now"]:
        if os.environ.get("PYTEST_XDIST_WORKER"):
            logger.warning(
                "Skipping `--delete-weights-now` cleanup under pytest-xdist to avoid cross-worker races."
            )
            return
        folder_to_delete = test_config["weights_dir"]
        if folder_to_delete.exists():
            shutil.rmtree(folder_to_delete)
            logger.info(f"Removed test artifacts from {folder_to_delete}")
