import sys

sys.path.append("./tools/yolov7/yolov7")

from models.experimental import attempt_load
from models.common import Conv
from models.yolo import Detect
from typing import Tuple
import onnx
import onnxsim
import torch

from tools.modules.exporter import Exporter


class YoloV7Exporter(Exporter):
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
            subtype="yolov7",
            output_names=["output1_yolov7", "output2_yolov7", "output3_yolov7"],
        )
        self.load_model()
    
    def load_model(self):
        # code based on export.py from YoloV5 repository
        # load the model
        model = attempt_load(self.model_path, map_location="cpu")
        # check num classes and labels
        assert model.nc == len(model.names), f'Model class count {model.nc} != len(names) {len(model.names)}'

        # check if image size is suitable
        gs = int(max(model.stride))  # grid size (max stride)
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

        m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
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

        inputs = conv_indices[-self.num_branches:]

        # Names of the node outputs you want to set as model outputs
        node_output_names = []
        for i, inp in enumerate(inputs):
            node_output_names.append(f"output{i+1}_yolov7")
            sigmoid = onnx.helper.make_node(
                'Sigmoid',
                inputs=[onnx_model.graph.node[inp].output[0]],
                outputs=[f'output{i+1}_yolov7'],
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
        self.make_nn_archive(self.model.names, self.model.nc)
