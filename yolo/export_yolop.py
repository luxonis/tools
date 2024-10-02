import sys

sys.path.append("./yolo/YOLOP")
import torch

from YOLOP.lib.models.common import Conv, Hardswish, Detect
from YOLOP.lib.models.YOLOP import MCnet, YOLOP

import torch.nn as nn
import onnx
from exporter import Exporter

DIR_TMP = "./tmp"


class YoloPExporter(Exporter):
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

        # code based on export_onnx.py from YOLOP repository
        # load the model
        model = MCnet(YOLOP)  # load FP32 model
        checkpoint = torch.load(
            self.weights_path.resolve(), map_location=torch.device("cpu")
        )
        model.load_state_dict(checkpoint["state_dict"])

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
        for _, m in model.named_modules():
            if isinstance(m, Conv):  # assign export-friendly activations
                if isinstance(m.act, nn.Hardswish):
                    m.act = Hardswish()
            elif isinstance(m, Detect):
                m.inplace = inplace
                m.onnx_dynamic = False
                if hasattr(m, "forward_export"):
                    m.forward = m.forward_export  # assign custom forward (optional)

        self.model = model

        m = (
            model.module.model[model.detector_index]
            if hasattr(model, "module")
            else model.model[model.detector_index]
        )
        self.num_branches = len(m.anchor_grid)

    def export_onnx(self):
        onnx_model, check = self.get_onnx()
        assert check, "assert check failed"

        det_idx = self.model.detector_index
        da_seg_idx = self.model.seg_out_idx[0]
        ll_seg_idx = self.model.seg_out_idx[1]

        # add named sigmoids for prunning in OpenVINO
        det_inputs = []
        seg_inputs = []
        for i, n in enumerate(onnx_model.graph.node):
            if n.name in [
                f"/model.{det_idx}/m.{i}/Conv" for i in range(self.num_branches)
            ]:
                det_inputs.append(i)
            if n.name == f"/model.{da_seg_idx}/act/Div":
                seg_inputs.append(i)
            if n.name == f"/model.{ll_seg_idx}/act/Div":
                seg_inputs.append(i)

        # Identify target nodes and outputs for detection and segmentation
        target_nodes_det = [onnx_model.graph.node[idx] for idx in det_inputs]
        target_outputs_det = []
        for node in target_nodes_det:
            target_outputs_det.extend(node.output)

        target_nodes_seg = [onnx_model.graph.node[idx] for idx in seg_inputs]
        target_outputs_seg = []
        for node in target_nodes_seg:
            target_outputs_seg.extend(node.output)

        # Clear the existing outputs
        onnx_model.graph.ClearField("output")

        nodes_to_remove = set()

        # Find dependent nodes recursively
        def find_dependent_nodes(output_names):
            dependent_nodes = []
            for node in onnx_model.graph.node:
                if any(input_name in output_names for input_name in node.input):
                    dependent_nodes.append(node)
            return dependent_nodes

        # Start with the initial target outputs and recursively find and remove dependent nodes
        current_outputs = target_outputs_det
        while current_outputs:
            dependent_nodes = find_dependent_nodes(current_outputs)
            if not dependent_nodes:
                break
            for node in dependent_nodes:
                nodes_to_remove.add(node.name)
            current_outputs = [
                output for node in dependent_nodes for output in node.output
            ]

        # Build a list of nodes to keep
        keep_nodes = [
            node for node in onnx_model.graph.node if node.name not in nodes_to_remove
        ]

        # Clear the graph's nodes and add back only the nodes we want to keep
        del onnx_model.graph.node[:]  # Delete all nodes from the original graph
        onnx_model.graph.node.extend(keep_nodes)  # Add back the nodes we want to keep

        # Find the corresponding ValueInfoProto for each target output
        value_info_map = {vi.name: vi for vi in onnx_model.graph.value_info}

        sigmoid_nodes = []

        # Create a sigmoid node for each target output and add its output to the graph's outputs
        for i, output in enumerate(target_outputs_det):
            sigmoid_output_name = (
                f"output{i+1}_yolop"  # Create a new output name for the sigmoid node
            )

            # Create a sigmoid node
            sigmoid_node = onnx.helper.make_node(
                "Sigmoid",
                inputs=[output],
                outputs=[sigmoid_output_name],
            )
            sigmoid_nodes.append(sigmoid_node)

            # Copy the shape information from the previous node's output (input to the Sigmoid)
            if output in value_info_map:
                output_value_info = value_info_map[output]
                sigmoid_output_info = onnx.helper.make_tensor_value_info(
                    sigmoid_output_name,
                    output_value_info.type.tensor_type.elem_type,
                    [
                        dim.dim_value
                        for dim in output_value_info.type.tensor_type.shape.dim
                    ],
                )
                onnx_model.graph.output.append(
                    sigmoid_output_info
                )  # Add the Sigmoid output as a graph output
            else:
                raise ValueError(f"ValueInfoProto not found for {output}")

        # Append the sigmoid nodes directly to the graph
        onnx_model.graph.node.extend(sigmoid_nodes)

        # Add the segmentation outputs to the graph's outputs (after being removed previously)
        for i, output in enumerate(target_outputs_seg):
            if i == 0:
                new_output_name = "drive_area_seg_masks"
            else:
                new_output_name = "lane_line_seg_masks"

            node_name = [n.name for n in onnx_model.graph.node if output in n.input][0]
            for node in onnx_model.graph.node:
                if node.name == node_name:
                    node.output[0] = new_output_name

                    output_value_info = value_info_map[output]
                    tensor_info = onnx.helper.make_tensor_value_info(
                        new_output_name,
                        output_value_info.type.tensor_type.elem_type,
                        [
                            dim.dim_value
                            for dim in output_value_info.type.tensor_type.shape.dim
                        ],
                    )
                    onnx_model.graph.output.append(tensor_info)
                    break

        onnx.checker.check_model(onnx_model)  # check onnx model

        # save the simplified model
        self.f_simplified = (
            self.conv_path / f"{self.model_name}-simplified.onnx"
        ).resolve()
        onnx.save(onnx_model, self.f_simplified)
        return self.f_simplified

    def export_json(self):
        # generate anchors and sides
        anchors, sides = [], []
        m = (
            self.model.module.model[self.model.detector_index]
            if hasattr(self.model, "module")
            else self.model.model[self.model.detector_index]
        )
        for i in range(self.num_branches):
            sides.append(m.anchor_grid[i].size()[3])
            for j in range(m.anchor_grid[i].size()[1]):
                anchors.extend(m.anchor_grid[i][0, j, 0, 0].numpy())
        anchors = [float(x) for x in anchors]

        # generate masks
        masks = dict()
        for i, num in enumerate(sides):
            masks[f"side{num}_{i}"] = list(range(i * 3, i * 3 + 3))

        return self.write_json(anchors, masks)
