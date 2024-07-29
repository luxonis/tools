import sys
sys.path.append("./yolo/ultralytics")

import re
import torch
import torch.nn as nn
import onnxsim
import onnx
from pathlib import Path
import json

from exporter import Exporter
from ultralytics.nn.tasks import temporary_modules, guess_model_task, Ensemble
from ultralytics.utils.checks import check_suffix, check_requirements
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, emojis
from ultralytics.nn.modules import Detect
from yolo.detect_head import DetectV10

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
    from ultralytics.utils.downloads import attempt_download_asset

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
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None  # combined args
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = args  # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        model.task = guess_model_task(model)
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        # Append
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())  # model in eval mode

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
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"Models differ in class counts {[m.nc for m in ensemble]}"
    return ensemble


class YoloV10Exporter(Exporter):

    def __init__(self, conv_path, weights_filename, imgsz, conv_id, n_shaves=6, use_legacy_frontend='false', use_rvc2='true'):
        super().__init__(conv_path, weights_filename, imgsz, conv_id, n_shaves, use_legacy_frontend, use_rvc2)
        self.load_model()
    
    def load_model(self):
        # load the model
        model = attempt_load_yolov10_weights(str(self.weights_path.resolve()), device="cpu", inplace=True, fuse=True)

        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        
        # check num classes and labels
        assert model.yaml["nc"] == len(names), f'Model class count {model.yaml["nc"]} != len(names) {len(names)}'

        # Replace with the custom Detection Head
        if isinstance(model.model[-1], (Detect)):
            model.model[-1] = DetectV10(model.model[-1], self.use_rvc2)

        self.num_branches = model.model[-1].nl

        # check if image size is suitable
        gs = max(int(model.stride.max()), 32)  # model stride
        if isinstance(self.imgsz, int):
            self.imgsz = [self.imgsz, self.imgsz]
        for sz in self.imgsz:
            if sz % gs != 0:
                raise ValueError(f"Image size is not a multiple of maximum stride {gs}")

        # ensure correct length
        if len(self.imgsz) != 2:
            raise ValueError(f"Image size must be of length 1 or 2.")

        model.eval()
        self.model = model

    def export_onnx(self):
        # export onnx model
        self.f_onnx = (self.conv_path / f"{self.model_name}.onnx").resolve()
        im = torch.zeros(1, 3, *self.imgsz[::-1])#.to(device)  # image size(1,3,320,192) BCHW iDetection
        torch.onnx.export(self.model, im, self.f_onnx, verbose=False, opset_version=12,
                        training=torch.onnx.TrainingMode.EVAL,
                        do_constant_folding=True,
                        input_names=['images'],
                        output_names=[f"output{i+1}_yolov6r2" for i in range(self.num_branches)],
                        dynamic_axes=None)

        # check if the arhcitecture is correct
        model_onnx = onnx.load(self.f_onnx)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # simplify the moodel
        onnx_model, check = onnxsim.simplify(model_onnx)
        assert check, 'assert check failed'
    
        onnx.checker.check_model(onnx_model)  # check onnx model

        # save the simplified model
        self.f_simplified = (self.conv_path / f"{self.model_name}-simplified.onnx").resolve()
        onnx.save(onnx_model, self.f_simplified)
        return self.f_simplified
    
    def export_openvino(self, version):
        super().export_openvino('v6r2')

        if not self.use_rvc2:
            # Replace opset8 with opset1 for Softmax layers
            # Read the content of the file
            with open(self.f_xml, 'r') as file:
                content = file.read()

            # Use the re.sub() function to replace the pattern with the new version
            new_content = re.sub(r'type="SoftMax" version="opset8"', 'type="SoftMax" version="opset1"', content)

            # Write the updated content back to the file
            with open(self.f_xml, 'w') as file:
                file.write(new_content)

        return self.f_xml, self.f_mapping, self.f_bin
    
    def write_json(self, anchors, masks, nc = None, names = None):
        # set parameters
        f = open((Path(__file__).parent / "json" / "yolo.json").resolve())
        content = json.load(f)

        content["model"]["xml"] = f"{self.model_name}.xml"
        content["model"]["bin"] = f"{self.model_name}.bin"
        content["nn_config"]["input_size"] = "x".join([str(x) for x in self.imgsz])
        if nc:
            content["nn_config"]["NN_specific_metadata"]["classes"] = nc
        else:
            content["nn_config"]["NN_specific_metadata"]["classes"] = self.model.nc
        content["nn_config"]["NN_specific_metadata"]["anchors"] = anchors
        content["nn_config"]["NN_specific_metadata"]["anchor_masks"] = masks
        content["nn_config"]["NN_specific_metadata"]["iou_threshold"] = 1.0
        if names:
            # use COCO labels if 80 classes, else use a placeholder
            content["mappings"]["labels"] = content["mappings"]["labels"] if nc == 80 else names
        else:
            content["mappings"]["labels"] = self.model.names if isinstance(self.model.names, list) else list(self.model.names.values())
        content["version"] = 1

        # save json
        f_json = (self.conv_path / f"{self.model_name}.json").resolve()
        with open(f_json, 'w') as outfile:
            json.dump(content, outfile, ensure_ascii=False, indent=4)

        self.f_json = f_json

        return self.f_json
    
    def export_json(self):
        # generate anchors and sides
        anchors, masks = [], {}

        nc = self.model.model[-1].nc
        names = [f"Class_{i}" for i in range(nc)]

        return self.write_json(anchors, masks, nc, names)
    

if __name__ == "__main__":
    # Test the YoloV10Exporter
    conv_path = Path(DIR_TMP)
    weights_filename = "yolov10n.pt"
    imgsz = 640
    conv_id = "test"
    nShaves = 6
    useLegacyFrontend = 'false'
    useRVC2 = 'false'
    exporter = YoloV10Exporter(conv_path, weights_filename, imgsz, conv_id, nShaves, useLegacyFrontend, useRVC2)
    exporter.export_onnx()
    exporter.export_openvino("v6r2")
    exporter.export_json()
