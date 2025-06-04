import sys

sys.path.append("./yolo/ultralytics")

import torch
import torch.nn as nn
from pathlib import Path

from ultralytics.nn.tasks import temporary_modules, guess_model_task, Ensemble
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, emojis
from ultralytics.utils.checks import check_suffix, check_requirements
from ultralytics.utils.downloads import attempt_download_asset
from ultralytics.nn.modules import Detect
from yolo.detect_head import DetectV10
from yolo.export_yolov8 import YoloV8Exporter


DIR_TMP = "./tmhttps://docs.luxonis.com/en/latest/_static/logo.pngp"


def torch_safe_load(weight):
    """
    This function attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised,
    it catches the error, logs a warning message, and attempts to install the missing module via the
    check_requirements() function. After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.

    Returns:
        (dict): The loaded PyTorch model.
    """
    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
            },
        ):
            ckpt = torch.load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'"
                )
            ) from e
        LOGGER.warning(
            f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'"
        )
        check_requirements(e.name)  # install missing module
        ckpt = torch.load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"WARNING ⚠️ The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file  # load


def attempt_load_yolov10_weights(weights, device=None, inplace=True, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""

    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # load ckpt
        args = (
            {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None
        )  # combined args
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = args  # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        model.task = guess_model_task(model)
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        # Append
        ensemble.append(
            model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()
        )  # model in eval mode

    # Module updates
    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    LOGGER.info(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[
        int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))
    ].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), (
        f"Models differ in class counts {[m.nc for m in ensemble]}"
    )
    return ensemble


class YoloV10Exporter(YoloV8Exporter):
    def __init__(
        self,
        conv_path: str,
        weights_filename: str,
        imgsz: tuple[int, int],
        conv_id: str,
        n_shaves: int = 6,
        use_legacy_frontend: bool = False,
        use_rvc2: bool = True,
    ):
        super().__init__(
            conv_path=conv_path,
            weights_filename=weights_filename,
            imgsz=imgsz,
            conv_id=conv_id,
            n_shaves=n_shaves,
            use_legacy_frontend=use_legacy_frontend,
            use_rvc2=use_rvc2,
        )
        self.load_model()

    def load_model(self):
        # load the model
        model = attempt_load_yolov10_weights(
            str(self.weights_path.resolve()), device="cpu", inplace=True, fuse=True
        )

        names = (
            model.module.names if hasattr(model, "module") else model.names
        )  # get class names

        # check num classes and labels
        assert model.yaml["nc"] == len(names), (
            f"Model class count {model.yaml['nc']} != len(names) {len(names)}"
        )

        # Replace with the custom Detection Head
        if isinstance(model.model[-1], (Detect)):
            model.model[-1] = DetectV10(model.model[-1], self.use_rvc2)

        self.num_branches = model.model[-1].nl

        # check if image size is suitable
        gs = max(int(model.stride.max()), 32)  # model stride
        for sz in self.imgsz:
            if sz % gs != 0:
                raise ValueError(f"Image size is not a multiple of maximum stride {gs}")

        model.eval()
        self.model = model


if __name__ == "__main__":
    # Test the YoloV10Exporter
    conv_path = Path(DIR_TMP)
    weights_filename = "yolov10n.pt"
    imgsz = 640
    conv_id = "test"
    nShaves = 6
    useLegacyFrontend = "false"
    useRVC2 = "false"
    exporter = YoloV10Exporter(
        conv_path, weights_filename, imgsz, conv_id, nShaves, useLegacyFrontend, useRVC2
    )
    exporter.export_onnx()
    exporter.export_openvino("v6r2")
    exporter.export_json()
