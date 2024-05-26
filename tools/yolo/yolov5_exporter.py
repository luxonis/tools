import sys

sys.path.append("./tools/yolo/yolov5")
import onnx
import onnxsim
import torch.nn as nn
from models.common import Conv
from models.experimental import attempt_load
from models.yolo import Detect
from utils.activations import SiLU
import torch
from typing import Tuple

from tools.modules import Exporter


class YoloV5Exporter(Exporter):
    def __init__(
        self,
        model_path: str,
        imgsz: Tuple[int, int],
        use_rvc2: bool,
    ):
        super().__init__(
            model_path,
            imgsz,
            use_rvc2,
            subtype="yolov5",
            output_names=["output1_yolov5", "output2_yolov5", "output3_yolov5"],
        )
        self.load_model()

    def load_model(self):
        # code based on export.py from YoloV5 repository
        # load the model
        model = attempt_load(self.model_path, device="cpu")  # load FP32 model

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
        self.f_onnx = (self.output_folder / f"{self.model_name}.onnx").resolve()
        im = torch.zeros(1, 3, *self.imgsz[::-1])
        # export onnx model
        torch.onnx.export(
            self.model,
            im,
            self.f_onnx,
            verbose=False,
            opset_version=12,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=["images"],
            output_names=['output'],
            dynamic_axes=None,
        )

        # Check if the arhcitecture is correct
        onnx.checker.check_model(self.f_onnx)
        onnx_model = onnx.load(self.f_onnx)
        onnx_model, check = onnxsim.simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"

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
        # Save onnx model
        onnx.save(onnx_model, self.f_onnx)
        return self.f_onnx

    def export_nn_archive(self):
        """Export the model to NN archive format."""
        self.make_nn_archive(list(self.model.names.values()), self.model.nc)
