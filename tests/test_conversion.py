import os
import requests
import pytest
from uuid import uuid4
import urllib.parse
import logging

from constants import (
    MODEL_TYPE2URL,
    DEFAULT_NSHAVES,
    DEFAULT_USE_LEGACY_FRONTEND,
    TEST_MODELS,
    STATUS_OK,
)

# Logging
LOG_FILE = "conversion_test.log"

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


def convert_model(model_path: str, version: str, url: str, shape: int, use_rvc2: bool):
    logger.info(
        f"Converting model: {model_path}, version={version}, shape={shape}, RVC2={use_rvc2}, URL={url}"
    )
    request_url = url
    if version == "v7":
        request_url = urllib.parse.urljoin(url, "yolov7/")
    elif version == "v6":
        request_url = urllib.parse.urljoin(url, "yolov6r1/")
    elif version == "v6r2":
        request_url = urllib.parse.urljoin(url, "yolov6r3/")
    request_url = urllib.parse.urljoin(request_url, "upload")

    with open(model_path, "rb") as file:
        response = requests.post(
            url=request_url,
            files={"file": file},
            data={
                "version": version,
                "inputshape": str(shape),
                "id": str(uuid4()),
                "nShaves": DEFAULT_NSHAVES,
                "useLegacyFrontend": DEFAULT_USE_LEGACY_FRONTEND,
                "useRVC2": use_rvc2,
            },
            stream=True,
        )
    out_file = f"converted_{os.path.basename(model_path).replace('.pt', '')}{'_rvc3' if not use_rvc2 else ''}.zip"
    with open(out_file, "wb") as f:
        for chunk in response.iter_content(8192):
            f.write(chunk)

    logger.debug(f"Conversion complete. Output saved to {out_file}")
    return response.status_code, out_file


@pytest.mark.parametrize(
    "model",
    TEST_MODELS,
    ids=[
        f"{model['name']}_{'rvc2' if model['use_rvc2'] else 'rvc3'}"
        for model in TEST_MODELS
    ],
)
def test_yolo_conversion(model, test_config):
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

    url = test_config["url"]
    final_url = urllib.parse.urljoin(url, model["url_suffix"])

    download_weights = test_config["download_weights"]
    delete_output = test_config["delete_output"]

    if test_config["is_local"]:
        save_folder = "./weights/"
    else:
        save_folder = model["folder"]

    model_path = os.path.join(save_folder, f"{model['name']}.pt")
    if not os.path.exists(model_path):
        if download_weights:
            model_path = download_model(
                model["name"],
                save_folder,
            )
        else:
            pytest.skip("Weights not present and `download_weights` not set")

    status_code, output_file = convert_model(
        model_path,
        version=model["version"],
        url=final_url,
        shape=model["shape"],
        use_rvc2=model["use_rvc2"],
    )

    assert os.path.exists(output_file)

    try:
        if delete_output and os.path.exists(output_file):
            os.remove(output_file)
            logger.debug(f"Removing {output_file}")
    except Exception as e:
        logger.error(f"{e}")

    logger.info(f"Conversion status: {status_code}")
    assert status_code == STATUS_OK
