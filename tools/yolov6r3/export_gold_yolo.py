import sys

sys.path.append("./Efficient-Computing/Detection/Gold-YOLO/")
sys.path.append("./Efficient-Computing/Detection/Gold-YOLO/gold_yolo/")
sys.path.append("./Efficient-Computing/Detection/Gold-YOLO/yolov6/utils/")

import onnx
import onnxsim
import torch
from checkpoint import load_checkpoint as load_checkpoint_gold_yolo
from switch_tool import switch_to_deploy

from tools.modules.exporter import Exporter
from yolo.detect_head import DetectV6R3


class GoldYoloExporter(Exporter):
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
        # Load the model
        model = load_checkpoint_gold_yolo(
            str(self.weights_path.resolve()), map_location="cpu"
        )

        model.detect = DetectV6R3(model.detect, self.use_rvc2)
        self.num_branches = len(model.detect.grid)

        # switch to deploy
        model = switch_to_deploy(model)

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

    def export_json(self):
        # generate anchors and sides
        anchors, masks = [], {}

        nc = self.model.detect.nc
        names = [f"Class_{i}" for i in range(nc)]

        return self.write_json(anchors, masks, nc, names)
