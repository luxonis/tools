from typing import Any
from pathlib import Path
import onnx

from export_yolov5 import YoloV5Exporter


class YoloV5SegExporter(YoloV5Exporter):
    def __init__(self, conv_path: Path, weights_filename: str, imgsz: int,
                conv_id: Any, n_shaves: int = 6, use_legacy_frontend: str = 'false'):
        super().__init__(conv_path, weights_filename, imgsz, conv_id, n_shaves, use_legacy_frontend)

    # load the model
    def load_model(self):
        super().load_model()
        m = self.model.module.model[-1] if hasattr(self.model, 'module') else self.model.model[-1]

        self.num_masks = m.nm           # number of masks
        self.num_out = m.no             # number of outputs per anchor
        self.num_anch = m.na            # number of anchors
        self.num_det_out = m.no-m.nm    # number of outputs without masks

    # splitting layers into masks and the detection output
    # using sigmoid on the detections, concatenating
    def prune_model(self, onnx_model: onnx.ModelProto, inputs: list):
        for i, inp in enumerate(inputs):
            slice_out = onnx.helper.make_node(
                'Split',
                inputs=[inp],
                outputs=[f'split_outputs{i+1}', f'split_masks{i+1}'],
                split=[self.num_det_out, self.num_masks],
                axis=2
            )

            sigmoid_out = onnx.helper.make_node(
                'Sigmoid',
                inputs=[f'split_outputs{i+1}'],
                outputs=[f'sigmoid_outputs{i+1}']
            )

            out_node = onnx.helper.make_node(
                'Concat',
                inputs=[f'sigmoid_outputs{i+1}', f'split_masks{i+1}'],
                outputs=[f'output{i+1}_yolov5seg'],
                axis=2
            )

            onnx_model.graph.node.append(slice_out)
            onnx_model.graph.node.append(sigmoid_out)
            onnx_model.graph.node.append(out_node)

    # exporting + preparing to prune the model 
    def export_onnx(self) -> Path:
        onnx_model, check = self.get_onnx()
        assert check, 'assert check failed'

        # get indices of convolustions (outputs), then getting connected reshapes
        # from reshapes it's much easier to split outputs
        conv_out = [n.output[0] for n in onnx_model.graph.node if "Conv" in n.name][-self.num_branches:]
        inputs = [n.output[0] for n in onnx_model.graph.node
                if "Reshape" in n.name and n.input[0] in conv_out]

        # preparing the model for pruning in OpenVINO
        self.prune_model(onnx_model, inputs)
        onnx.checker.check_model(onnx_model)  # check onnx model

        # save the simplified model
        self.f_simplified = (self.conv_path / f"{self.model_name}-simplified.onnx").resolve()
        onnx.save(onnx_model, self.f_simplified)
        return self.f_simplified
    
    # was facing an error without overriding the class, it should affect anything tho
    def export_openvino(self, version: str = 'v5seg'):
        return super().export_openvino(version)