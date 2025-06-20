import logging
import os
import shutil

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
        choices=["v5", "v6", "v6r2", "v6r4", "v7", "v8", "v9", "v10", "v11"],
        default=None,
        help="If set then test only that specific yolo version",
    )
    parser.addoption(
        "--test-case",
        type=str,
        default=None,
        help="If set then test only that specific test case",
    )


@pytest.fixture(scope="session")
def test_config(pytestconfig):
    return {
        "download_weights": pytestconfig.getoption("download_weights"),
        "delete_output": not pytestconfig.getoption("no_delete_output"),
        "yolo_version": pytestconfig.getoption("yolo_version"),
        "test_case": pytestconfig.getoption("test_case"),
    }


@pytest.fixture(scope="function", autouse=True)
def cleanup_after_tests(test_config):
    yield  # Tests run here
    if test_config["delete_output"]:
        folder_to_delete = "shared_with_container"
        if os.path.exists(folder_to_delete):
            shutil.rmtree(folder_to_delete)
            logger.info(f"Removed test artifacts from {folder_to_delete}")
