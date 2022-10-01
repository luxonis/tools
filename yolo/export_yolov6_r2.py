import sys

from yolo.detect_head import DetectV2
# sys.path.append("./yolo/")

# try:
#     sys.path.remove("./yolo/YOLOv6R1")
#     # sys.path.remove("./yolo/newer/YOLOv6R2/yolov6")
# except:
#     pass

# sys.path.append("./yolo/newer/YOLOv6R2")
sys.path.append("./yolo/YOLOv6R2")
# sys.path.append("./yolo/newer/YOLOv6R2/yolov6")

import torch
import torch.nn as nn

from yolov6.models.yolo import *
from yolov6.layers.common import *

from yolov6.layers.common import RepVGGBlock
from yolov6.utils.checkpoint import load_checkpoint
import onnx
from exporter import Exporter

DIR_TMP = "./tmp"

import numpy as np
import onnxsim
import subprocess


# class YoloV6R2(nn.Module):
#     def __init__(self, weights, input_shape):
#         super().__init__()
#         self.model = load_checkpoint(weights, map_location="cpu", inplace=True, fuse=True)  # load FP32 model
#         self.model.eval()
#         self.stride = self.model.stride

#         self.input_shape = input_shape
#         self.grids = []
#         self.total_shape = 0
#         self.channel_shapes = []
#         self.boundaries = []
#         if isinstance(self.input_shape, int):
#             for channel_shape in [self.input_shape//8, self.input_shape//16, self.input_shape//32]:
#                 self.channel_shapes.append((channel_shape, channel_shape))
#                 self.total_shape += channel_shape*channel_shape
#                 self.boundaries.append(self.total_shape)
#                 yv, xv = torch.meshgrid([torch.arange(channel_shape), torch.arange(channel_shape)])
#                 self.grids.append(torch.stack((xv, yv), 2).view(1, channel_shape, channel_shape, 2).float())
#         elif isinstance(self.input_shape, list) and len(self.input_shape) == 2:
#             self.input_shape = np.array(self.input_shape)

#             for channel_shape_x, channel_shape_y in [self.input_shape//8, self.input_shape//16, self.input_shape//32]:
#                 self.channel_shapes.append((channel_shape_y, channel_shape_x))
#                 self.total_shape += channel_shape_x*channel_shape_y
#                 self.boundaries.append(self.total_shape)
#                 yv, xv = torch.meshgrid([torch.arange(channel_shape_y), torch.arange(channel_shape_x)])
#                 self.grids.append(torch.stack((xv, yv), 2).view(1, channel_shape_y, channel_shape_x, 2).float())
        
#         for layer in self.model.modules():
#             if isinstance(layer, RepVGGBlock):
#                 layer.switch_to_deploy()

#     def forward(self, im, val=False):
#         y = self.model(im)[0]
#         if y.shape[0] == self.boundaries[2]:
#             y = y.unsqueeze(0)
#         if isinstance(y, np.ndarray):
#             y = torch.tensor(y)
#         result1, result2, result3 = y[0, :self.boundaries[0]].view((1, *self.channel_shapes[0], -1)), y[0, self.boundaries[0]:self.boundaries[1]].view((1, *self.channel_shapes[1], -1)), y[0, self.boundaries[1]:].view((1, *self.channel_shapes[2], -1))
#         # result1, result2, result3 = y[0, :6400].view((1, 80, 80, -1)), y[0, 6400:8000].view((1, 40, 40, -1)), y[0, 8000:].view((1, 20, 20, -1))
        
#         def inverse_opperations(channel, stride, grid):
#             # _, ny, nx, _ = channel.shape
#             # yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
#             # xy = (channel[..., 0:2]/stride) - torch.stack((xv, yv), 2).view(1, ny, nx, 2).float()
#             xy = (channel[..., 0:2]/stride) - grid
#             wh = torch.log(channel[..., 2:4]/stride)
#             return torch.cat((xy, wh, channel[..., 4:]), -1)
        
#         output1 = inverse_opperations(result1, self.stride[0], self.grids[0])
#         output2 = inverse_opperations(result2, self.stride[1], self.grids[1])
#         output3 = inverse_opperations(result3, self.stride[2], self.grids[2])

#         return output1.permute(0, 3, 1, 2), output2.permute(0, 3, 1, 2), output3.permute(0, 3, 1, 2)


class YoloV6R2Exporter(Exporter):

    def __init__(self, conv_path, weights_filename, imgsz, conv_id):
        super().__init__(conv_path, weights_filename, imgsz, conv_id)
        self.input_shape = imgsz
        self.load_model()

    def load_model(self):
        # load the model
        # model = YoloV6R2(weights=str(self.weights_path.resolve()), input_shape=self.input_shape)
        model = load_checkpoint(str(self.weights_path.resolve()), map_location="cpu", inplace=True, fuse=True)  # load FP32 model
        print(model.detect.cls_convs[0].conv.weight[0, 0])
        # print(model.detect.cls_convs[0].conv.weight.shape) # torch.Size([32, 32, 3, 3])
        model.detect = DetectV2(model.detect)
        print(model.detect.cls_convs[0].conv.weight[0, 0])
        
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()

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

    def get_onnx(self):
        # export onnx model
        self.f_onnx = (self.conv_path / f"{self.model_name}.onnx").resolve()
        im = torch.zeros(1, 3, *self.imgsz[::-1])#.to(device)  # image size(1,3,320,192) BCHW iDetection
        torch.onnx.export(self.model, im, self.f_onnx, verbose=False, opset_version=12,
                        training=torch.onnx.TrainingMode.EVAL,
                        do_constant_folding=True,
                        input_names=['images'],
                        output_names=['output1_yolov6', 'output2_yolov6', 'output3_yolov6'],
                        dynamic_axes=None)

        # check if the arhcitecture is correct
        model_onnx = onnx.load(self.f_onnx)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # simplify the moodel
        return onnxsim.simplify(model_onnx)
    
    def export_onnx(self):
        onnx_model, check = self.get_onnx()
        assert check, 'assert check failed'
        
        onnx.checker.check_model(onnx_model)  # check onnx model

        # save the simplified model
        self.f_simplified = (self.conv_path / f"{self.model_name}-simplified.onnx").resolve()
        onnx.save(onnx_model, self.f_simplified)
        return self.f_simplified

    def export_openvino(self, version):

        if self.f_simplified is None:
            self.export_onnx()
        
        # export to OpenVINO and prune the model in the process
        cmd = f"mo --input_model '{self.f_simplified}' " \
        f"--output_dir '{self.conv_path.resolve()}' " \
        f"--model_name '{self.model_name}' " \
        '--data_type FP16 ' \
        '--reverse_input_channel ' \
        '--scale 255 '

        try:
            subprocess.check_output(cmd, shell=True)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError()

        # set paths
        self.f_xml = (self.conv_path / f"{self.model_name}.xml").resolve()
        self.f_bin = (self.conv_path / f"{self.model_name}.bin").resolve()
        self.f_mapping = (self.conv_path / f"{self.model_name}.mapping").resolve()

        return self.f_xml, self.f_mapping, self.f_bin

    def export_json(self):
        # generate anchors and sides
        anchors, masks = [], {}

        nc = self.model.detect.nc
        names = [f"Class_{i}" for i in range(nc)]

        return self.write_json(anchors, masks, nc, names)