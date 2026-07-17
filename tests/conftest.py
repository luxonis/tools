import logging
import os
import shutil
from pathlib import Path

import pytest

pytest_plugins = ["e2e_shards"]

logger = logging.getLogger()

os.environ.setdefault("LUXONIS_TELEMETRY_ENABLED", "false")


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
    }


@pytest.fixture(scope="session")
def test_workspace(tmp_path_factory, request) -> Path:
    """Create one isolated conversion workspace per pytest worker."""
    worker_id = getattr(request.config, "workerinput", {}).get("workerid", "master")
    workspace = tmp_path_factory.mktemp(f"tools-tests-{worker_id}")
    (workspace / "weights").mkdir(exist_ok=True)
    return workspace


def _artifact_path(request: pytest.FixtureRequest, name: str) -> Path:
    """Resolve cleanup paths without changing E2E behavior."""
    if "test_workspace" in request.fixturenames:
        workspace = request.getfixturevalue("test_workspace")
        return Path(workspace) / name

    return Path(name)


@pytest.fixture(scope="function", autouse=True)
def cleanup_output_after_tests(test_config, request):
    yield  # Tests run here
    if test_config["delete_output"]:
        folder_to_delete = _artifact_path(
            request,
            "shared_with_container",
        )
        if folder_to_delete.exists():
            shutil.rmtree(folder_to_delete)
            logger.info(f"Removed test artifacts from {folder_to_delete}")


@pytest.fixture(scope="function", autouse=True)
def cleanup_weights_after_tests(test_config, request):
    yield  # Tests run here
    if test_config["delete_weights_now"]:
        folder_to_delete = _artifact_path(request, "weights")
        if folder_to_delete.exists():
            shutil.rmtree(folder_to_delete)
            logger.info(f"Removed test artifacts from {folder_to_delete}")
