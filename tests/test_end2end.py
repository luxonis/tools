from __future__ import annotations

import logging
import os
import subprocess

import pytest
from constants import SAVE_FOLDER, TEST_MODELS
from helper_functions import download_model, nn_archive_checker

logger = logging.getLogger()
logger.setLevel(logging.INFO)


@pytest.mark.parametrize(
    "model",
    TEST_MODELS,
    ids=[model["name"] for model in TEST_MODELS],
)
def test_cli_conversion(model: dict, test_config: dict, subtests):
    """Tests the whole CLI conversion flow with no extra params specified."""
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
