import sys

sys.path.append("./yolov5")
import onnx
import torch
import torch.nn as nn
from yolov5.models.common import Conv
from yolov5.models.experimental import attempt_load
from yolov5.models.yolo import Detect
from yolov5.utils.activations import SiLU

from tools.modules.exporter import Exporter

DIR_TMP = "./tmp"


class YoloV5Exporter(Exporter):
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
        model = attempt_load(
            self.weights_path.resolve(), device=torch.device("cpu")
        )  # load FP32 model

        # check num classes and labels
        assert model.nc == len(
            model.names
        ), f"Model class count {model.nc} != len(names) {len(model.names)}"

        # check if image size is suitable
        gs = int(max(model.stride))  # grid size (max stride)
        if isinstance(self.imgsz, int):
            self.imgsz = [self.imgsz, self.imgsz]
        for sz in self.imgsz:
            if sz % gs != 0:
                raise ValueError(f"Image size is not a multiple of maximum stride {gs}")

        # ensure correct length
        if len(self.imgsz) != 2:
            raise ValueError("Image size must be of length 1 or 2.")

        inplace = True

        model.eval()
        for k, m in model.named_modules():
            if isinstance(m, Conv):  # assign export-friendly activations
                if isinstance(m.act, nn.SiLU):
                    m.act = SiLU()
            elif isinstance(m, Detect):
                m.inplace = inplace
                m.onnx_dynamic = False
                if hasattr(m, "forward_export"):
                    m.forward = m.forward_export  # assign custom forward (optional)

        self.model = model

        m = model.module.model[-1] if hasattr(model, "module") else model.model[-1]
        self.num_branches = len(m.anchor_grid)

    def export_onnx(self):
        onnx_model, check = self.get_onnx()
        assert check, "assert check failed"

        # add named sigmoids for prunning in OpenVINO
        conv_indices = []
        for i, n in enumerate(onnx_model.graph.node):
            if "Conv" in n.name:
                conv_indices.append(i)

        inputs = conv_indices[-self.num_branches :]

        # Names of the node outputs you want to set as model outputs
        node_output_names = []
        for i, inp in enumerate(inputs):
            node_output_names.append(f"output{i+1}_yolov5")
            sigmoid = onnx.helper.make_node(
                "Sigmoid",
                inputs=[onnx_model.graph.node[inp].output[0]],
                outputs=[f"output{i+1}_yolov5"],
            )
            onnx_model.graph.node.append(sigmoid)

        # Create ValueInfoProto messages for the desired outputs
        output_value_infos = []
        for output_name in node_output_names:
            for node in onnx_model.graph.node:
                if output_name in node.output:
                    # Copy information from the output's ValueInfoProto
                    # (this includes the data type and shape, if they are known)
                    for value_info in onnx_model.graph.value_info:
                        if value_info.name == output_name:
                            output_value_infos.append(value_info)

        # Clear the existing outputs
        onnx_model.graph.ClearField("output")

        # Add the new outputs
        for value_info in output_value_infos:
            onnx_model.graph.output.add().CopyFrom(value_info)

        onnx.checker.check_model(onnx_model)  # check onnx model

        return self.f_simplified
