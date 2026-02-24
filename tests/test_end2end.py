from __future__ import annotations

import logging
import os
import subprocess

import pytest
from constants import PRIVATE_TEST_MODELS, SAVE_FOLDER, TEST_MODELS
from helper_functions import (
    download_model,
    download_private_model,
    load_latest_nn_archive_config,
    nn_archive_checker,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

N_VARIANT_OUTPUT_NAME_CHECKS = [
    {
        "name": "yolov8n",
        "version": "v8",
        "model_outputs": [
            "output1_yolov6r2",
            "output2_yolov6r2",
            "output3_yolov6r2",
        ],
        "head_outputs": [
            "output1_yolov6r2",
            "output2_yolov6r2",
            "output3_yolov6r2",
        ],
        "yolo_outputs": [
            "output1_yolov6r2",
            "output2_yolov6r2",
            "output3_yolov6r2",
        ],
    },
    {
        "name": "yolov8n-seg",
        "version": "v8",
        "model_outputs": [
            "output1_yolov8",
            "output2_yolov8",
            "output3_yolov8",
            "output1_masks",
            "output2_masks",
            "output3_masks",
            "protos_output",
        ],
        "head_outputs": [
            "output1_yolov8",
            "output2_yolov8",
            "output3_yolov8",
            "output1_masks",
            "output2_masks",
            "output3_masks",
            "protos_output",
        ],
        "yolo_outputs": ["output1_yolov8", "output2_yolov8", "output3_yolov8"],
        "mask_outputs": ["output1_masks", "output2_masks", "output3_masks"],
    },
    {
        "name": "yolov8n-pose",
        "version": "v8",
        "model_outputs": [
            "output1_yolov8",
            "output2_yolov8",
            "output3_yolov8",
            "kpt_output1",
            "kpt_output2",
            "kpt_output3",
        ],
        "head_outputs": [
            "output1_yolov8",
            "output2_yolov8",
            "output3_yolov8",
            "kpt_output1",
            "kpt_output2",
            "kpt_output3",
        ],
        "yolo_outputs": ["output1_yolov8", "output2_yolov8", "output3_yolov8"],
        "keypoints_outputs": ["kpt_output1", "kpt_output2", "kpt_output3"],
    },
    {
        "name": "yolo26n",
        "version": "v26",
        "model_outputs": ["output_yolo26"],
        "head_outputs": ["output_yolo26"],
        "yolo_outputs": ["output_yolo26"],
    },
    {
        "name": "yolo26n-seg",
        "version": "v26",
        "model_outputs": ["output_yolo26", "output_masks", "protos_output"],
        "head_outputs": ["output_yolo26", "output_masks", "protos_output"],
        "yolo_outputs": ["output_yolo26"],
        "mask_outputs": ["output_masks"],
    },
    {
        "name": "yolo26n-pose",
        "version": "v26",
        "model_outputs": ["output_yolo26", "kpt_output"],
        "head_outputs": ["output_yolo26", "kpt_output"],
        "yolo_outputs": ["output_yolo26"],
        "keypoints_outputs": ["kpt_output"],
    },
]


@pytest.mark.parametrize(
    "model",
    TEST_MODELS,
    ids=[
        model.get("cli_version", model["name"])
        if model.get("cli_version")
        else model["name"]
        for model in TEST_MODELS
    ],
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
            pytest.skip("Weights missing and `download_weights` not set")

    command = ["tools", model_path]
    if model.get("cli_version"):
        command += ["--version", model.get("cli_version")]
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


@pytest.mark.parametrize(
    "model_case",
    N_VARIANT_OUTPUT_NAME_CHECKS,
    ids=[model_case["name"] for model_case in N_VARIANT_OUTPUT_NAME_CHECKS],
)
def test_n_variant_nnarchive_outputs(model_case: dict, test_config: dict):
    """Checks NNArchive output-related fields for YOLOv8n and YOLO26n model variants."""
    if (
        test_config["test_case"] is not None
        and model_case["name"] != test_config["test_case"]
    ):
        pytest.skip(
            f"Test case ({model_case['name']}) doesn't match selected test case ({test_config['test_case']})"
        )

    if (
        test_config["yolo_version"] is not None
        and model_case["version"] != test_config["yolo_version"]
    ):
        pytest.skip(
            f"Model version ({model_case['version']}) doesn't match selected version ({test_config['yolo_version']})."
        )

    model_path = os.path.join(SAVE_FOLDER, f"{model_case['name']}.pt")
    if not os.path.exists(model_path):
        if test_config["download_weights"]:
            model_path = download_model(model_case["name"], SAVE_FOLDER)
        else:
            pytest.skip("Weights missing and `download_weights` not set")

    command = ["tools", model_path]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if result.returncode != 0:
        pytest.fail(f"Exit code: {result.returncode}, Output: {result.stdout}")

    cfg = load_latest_nn_archive_config()
    output_names = [output["name"] for output in cfg["model"]["outputs"]]
    head = cfg["model"]["heads"][0]
    metadata = head["metadata"]
    head_output_names = head["outputs"]
    yolo_output_names = metadata["yolo_outputs"] or []
    mask_output_names = metadata["mask_outputs"] or []
    keypoint_output_names = metadata["keypoints_outputs"] or []

    for key, actual in [
        ("model_outputs", output_names),
        ("head_outputs", head_output_names),
        ("yolo_outputs", yolo_output_names),
        ("mask_outputs", mask_output_names),
        ("keypoints_outputs", keypoint_output_names),
    ]:
        for expected_name in model_case.get(key, []):
            assert expected_name in actual, (
                f"{key}: expected `{expected_name}` for {model_case['name']}, got {actual}"
            )


@pytest.mark.parametrize(
    "model",
    PRIVATE_TEST_MODELS,
    ids=[model["name"] for model in PRIVATE_TEST_MODELS],
)
def test_private_model_conversion(model: dict, test_config: dict, subtests):
    """Tests the CLI conversion flow for models that are not part of the standard
    pretrained suite.

    Hosted on GCP bucket.
    """
    if not test_config["test_private"]:
        pytest.skip("Private model tests not enabled (use --test-private flag)")

    logger.info(f"Testing private model: {model['name']}")

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

    model_path = os.path.join(SAVE_FOLDER, f"{model['name']}.pt")
    if not os.path.exists(model_path):
        model_path = download_private_model(
            model["name"],
            model["filename"],
            SAVE_FOLDER,
        )

    command = ["tools", model_path]
    if model.get("size"):
        command += ["--imgsz", model.get("size")]

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
