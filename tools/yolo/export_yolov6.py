import sys

sys.path.append("./YOLOv6")

import onnx
import onnxsim
import torch
from yolov6.layers.common import RepVGGBlock
from yolov6.models.heads.effidehead_distill_ns import Detect
from yolov6.utils.checkpoint import load_checkpoint

from tools.modules.exporter import Exporter
from yolo.detect_head import DetectV6R4m, DetectV6R4s

DIR_TMP = "./tmp"


class YoloV6R4Exporter(Exporter):
    def __init__(
        self,
        conv_path,
        weights_filename,
        imgsz,
        conv_id,
        n_shaves=6,
        use_legacy_frontend="false",
        use_rvc2="true",
    ):
        super().__init__(
            conv_path,
            weights_filename,
            imgsz,
            conv_id,
            n_shaves,
            use_legacy_frontend,
            use_rvc2,
        )
        self.load_model()

    def load_model(self):
        # code based on export.py from YoloV5 repository
        # load the model
        model = load_checkpoint(
            str(self.weights_path.resolve()),
            map_location="cpu",
            inplace=True,
            fuse=True,
        )  # load FP32 model

        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()

        if isinstance(model.detect, Detect):
            model.detect = DetectV6R4s(model.detect, self.use_rvc2)
        else:
            model.detect = DetectV6R4m(model.detect, self.use_rvc2)

        self.num_branches = len(model.detect.grid)

        # check if image size is suitable
        gs = 2 ** (2 + self.num_branches)  # 1 = 8, 2 = 16, 3 = 32
        if isinstance(self.imgsz, int):
            self.imgsz = [self.imgsz, self.imgsz]
        for sz in self.imgsz:
            if sz % gs != 0:
                raise ValueError(f"Image size is not a multiple of maximum stride {gs}")

        # ensure correct length
        if len(self.imgsz) != 2:
            raise ValueError("Image size must be of length 1 or 2.")

        model.eval()
        self.model = model

    def export_onnx(self):
        # export onnx model
        self.f_onnx = (self.conv_path / f"{self.model_name}.onnx").resolve()
        im = torch.zeros(
            1, 3, *self.imgsz[::-1]
        )  # .to(device)  # image size(1,3,320,192) BCHW iDetection
        torch.onnx.export(
            self.model,
            im,
            self.f_onnx,
            verbose=False,
            opset_version=12,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["output1_yolov6r2", "output2_yolov6r2", "output3_yolov6r2"],
            dynamic_axes=None,
        )

        # check if the arhcitecture is correct
        model_onnx = onnx.load(self.f_onnx)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # simplify the moodel

        onnx_model, check = onnxsim.simplify(model_onnx)
        assert check, "assert check failed"

        onnx.checker.check_model(onnx_model)  # check onnx model

        # save the simplified model
        self.f_simplified = (
            self.conv_path / f"{self.model_name}-simplified.onnx"
        ).resolve()
        onnx.save(onnx_model, self.f_simplified)
        return self.f_simplified
