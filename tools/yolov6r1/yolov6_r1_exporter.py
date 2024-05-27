import sys

sys.path.append("./tools/yolov6r1/YOLOv6R1")

import onnx
import onnxsim
from yolov6.layers.common import RepVGGBlock
from yolov6.utils.checkpoint import load_checkpoint
import torch
from typing import Tuple

from tools.modules import Exporter, DetectV6R1


class YoloV6R1Exporter(Exporter):
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
            subtype="yolov6",
            output_names=["output1_yolov6", "output2_yolov6", "output3_yolov6"],
        )
        self.load_model()
    
    def load_model(self):
        # code based on export.py from YoloV5 repository
        # load the model
        model = load_checkpoint(self.model_path, map_location="cpu", inplace=True, fuse=True)  # load FP32 model
        
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
        
        if hasattr(model.detect, 'obj_preds'):
            model.detect = DetectV6R1(model.detect)
        else:
            raise ValueError(f"Error while loading model (This may be caused by trying to convert either the latest release 4.0 that isn't supported yet, or by releases 2.0 or 3.0, in which case, try to convert using the 'YoloV6 (R2, R3)' option).")
        
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
            raise ValueError(f"Image size must be of length 1 or 2.")

        model.eval()
        self.model = model

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

        # get contacts ready for parsing
        conc_idx = []
        for i, n in enumerate(onnx_model.graph.node):
            if "Concat" in n.name:
                conc_idx.append(i)

        outputs = conc_idx[-(self.num_branches+1):-1]
        change_inputs = []

        for i, idx in enumerate(outputs):
            onnx_model.graph.node[idx].name = f"output{i+1}_yolov6"
            change_inputs.append(onnx_model.graph.node[idx].output[0])
            onnx_model.graph.node[idx].output[0] = f"output{i+1}_yolov6"
        
        for i in range(len(onnx_model.graph.node)):
            for j in range(len(onnx_model.graph.node[i].input)):
                if onnx_model.graph.node[i].input[j] in change_inputs:
                    output_i = change_inputs.index(onnx_model.graph.node[i].input[j])+1
                    onnx_model.graph.node[i].input[j] = f"output{output_i}_yolov6"
        
        onnx.checker.check_model(onnx_model)  # check onnx model
        # Save onnx model
        onnx.save(onnx_model, self.f_onnx)
        return self.f_onnx
