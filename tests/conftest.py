import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--tools-url",
        type=str,
        default="https://tools.luxonis.com",
        help="Base URL for the tools service",
    )
    parser.addoption(
        "--download-weights",
        action="store_true",
        help="Download weights if not present",
    )
    parser.addoption(
        "--no-delete-output",
        action="store_true",
        help="Don't delete output zip files after test",
    )
    parser.addoption(
        "--is-local",
        action="store_true",
        help="If set then use ./weights/ for weights storing",
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
        "url": pytestconfig.getoption("tools_url"),
        "download_weights": pytestconfig.getoption("download_weights"),
        "delete_output": not pytestconfig.getoption("no_delete_output"),
        "is_local": pytestconfig.getoption("is_local"),
        "yolo_version": pytestconfig.getoption("yolo_version"),
        "test_case": pytestconfig.getoption("test_case"),
    }
