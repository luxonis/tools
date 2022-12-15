import sys

sys.path.append("./yolo/yolov7")

import torch
from yolov7.models.experimental import attempt_load
from yolov7.models.common import Conv
from yolov7.models.yolo import Detect
import torch.nn as nn
import onnx
from exporter import Exporter

DIR_TMP = "./tmp"

class YoloV7Exporter(Exporter):

    def __init__(self, conv_path, weights_filename, imgsz, conv_id):
        super().__init__(conv_path, weights_filename, imgsz, conv_id)
        
        self.load_model()
    
    def load_model(self):

        # code based on export.py from YoloV5 repository
        # load the model
        model = attempt_load(self.weights_path.resolve(), map_location=torch.device('cpu'))
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
        '''
        for k, m in model.named_modules():
            if isinstance(m, Conv):  # assign export-friendly activations
                if isinstance(m.act, nn.SiLU):
                    m.act = SiLU()
            elif isinstance(m, Detect):
                m.inplace = inplace
                m.onnx_dynamic = False
                if hasattr(m, 'forward_export'):
                    m.forward = m.forward_export  # assign custom forward (optional)
        '''

        self.model = model

        m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
        self.num_branches = len(m.anchor_grid)           

    def export_onnx(self):
        onnx_model, check = self.get_onnx()
        assert check, 'assert check failed'

        # add named sigmoids for prunning in OpenVINO
        conv_indices = []
        for i, n in enumerate(onnx_model.graph.node):
            if "Conv" in n.name:
                conv_indices.append(i)

        inputs = conv_indices[-self.num_branches:]

        for i, inp in enumerate(inputs):
            sigmoid = onnx.helper.make_node(
                'Sigmoid',
                inputs=[onnx_model.graph.node[inp].output[0]],
                outputs=[f'output{i+1}_yolov7'],
            )
            onnx_model.graph.node.append(sigmoid)

        onnx.checker.check_model(onnx_model)  # check onnx model

        # save the simplified model
        self.f_simplified = (self.conv_path / f"{self.model_name}-simplified.onnx").resolve()
        onnx.save(onnx_model, self.f_simplified)
        return self.f_simplified

    def export_json(self):
        # generate anchors and sides
        anchors, sides = [], []
        m = self.model.module.model[-1] if hasattr(self.model, 'module') else self.model.model[-1]
        for i in range(self.num_branches):
            sides.append(int(self.imgsz[0] // m.stride[i]))
            for j in range(m.anchor_grid[i].size()[1]):
                anchors.extend(m.anchor_grid[i][0, j, 0, 0].numpy())
        anchors = [float(x) for x in anchors]
        #sides.sort()

        # generate masks
        masks = dict()
        #for i, num in enumerate(sides[::-1]):
        for i, num in enumerate(sides):
            masks[f"side{num}"] = list(range(i*3, i*3+3))

        return self.write_json(anchors, masks)