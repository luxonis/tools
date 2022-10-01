import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys
sys.path.append("./yolo/YOLOv6R2")

from yolov6.layers.common import *
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox


class DetectV2(nn.Module):
    '''Efficient Decoupled Head
    With hardware-aware degisn, the decoupled head is optimized with
    hybridchannels methods.
    '''
    # def __init__(self, num_classes=80, anchors=1, num_layers=3, inplace=True, head_layers=None, use_dfl=True, reg_max=16):  # detection layer
    def __init__(self, old_detect):  # detection layer
        super().__init__()
        # self.nc = num_classes  # number of classes
        # self.no = num_classes + 5  # number of outputs per anchor
        # self.nl = num_layers  # number of detection layers
        # if isinstance(anchors, (list, tuple)):
        #     self.na = len(anchors[0]) // 2
        # else:
        #     self.na = anchors
        # self.anchors = anchors
        # self.grid = [torch.zeros(1)] * num_layers
        # self.prior_prob = 1e-2
        # self.inplace = inplace
        # stride = [8, 16, 32]  # strides computed during build
        # self.stride = torch.tensor(stride)
        # self.use_dfl = use_dfl
        # self.reg_max = reg_max
        # self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        # self.grid_cell_offset = 0.5
        # self.grid_cell_size = 5.0

        # # Init decouple head
        # self.stems = stems
        # self.cls_convs = cls_convs
        # self.reg_convs = reg_convs
        # self.cls_preds = cls_preds
        # self.reg_preds = reg_preds
        self.nc = old_detect.nc  # number of classes
        self.no = old_detect.no  # number of outputs per anchor
        self.nl = old_detect.nl  # number of detection layers
        self.anchors = old_detect.anchors
        self.grid = old_detect.grid # [torch.zeros(1)] * self.nl
        self.prior_prob = 1e-2
        self.inplace = old_detect.inplace
        stride = [8, 16, 32]  # strides computed during build
        self.stride = torch.tensor(stride)
        self.use_dfl = old_detect.use_dfl
        self.reg_max = old_detect.reg_max
        self.proj_conv = old_detect.proj_conv
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        # Init decouple head
        self.stems = old_detect.stems
        self.cls_convs = old_detect.cls_convs
        self.reg_convs = old_detect.reg_convs
        self.cls_preds = old_detect.cls_preds
        self.reg_preds = old_detect.reg_preds

    def forward(self, x):
        outputs = []
        for i in range(self.nl):
            b, _, h, w = x[i].shape
            l = h * w
            x[i] = self.stems[i](x[i])
            cls_x = x[i]
            reg_x = x[i]
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)
            
            if self.use_dfl:
                reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                reg_output = self.proj_conv(F.softmax(reg_output, dim=1))
            
            cls_output = torch.sigmoid(cls_output)
            conf, _ = cls_output.max(1, keepdim=True)
            output = torch.cat((reg_output, conf, cls_output), axis=1)
            # output = torch.cat((cls_output, conf, reg_output), axis=1)

            # cls_score_list.append(cls_output) #.reshape([b, self.nc, l]))
            # reg_dist_list.append(reg_output) #.reshape([b, 4, l]))
            # objectness_list.append(conf)
            # output = torch.cat((cls_output, conf), axis=1)
            # output = torch.cat((reg_output, output), axis=1)
            # output = torch.cat((reg_output, cls_output), axis=1)
            # output = torch.cat((conf, output), axis=1)

            # output = torch.cat((reg_output, cls_output), axis=1)
            # output = torch.cat((conf, output), axis=1)

            print(f'{i}. conf: {conf.shape}')
            print(f'{i}. output: {output.shape}')
            # outputs.append(output)
            outputs.append(torch.flip(output, [1]))
            # output = torch.zeros((b, self.no, h, w))
            # output[:, :4] = reg_output
            # output[:, 4] = conf
            # output[:, 5:] = cls_output
            # outputs.append(output)
        
        # cls_score_list = torch.cat(cls_score_list, axis=-1).permute(0, 2, 1)
        # reg_dist_list = torch.cat(reg_dist_list, axis=-1).permute(0, 2, 1)

        # print(f'reg_dist_list: {len(reg_dist_list)} 0: {reg_dist_list[0].shape} 1: {reg_dist_list[1].shape} 2: {reg_dist_list[2].shape}')
        # print(f'cls_score_list: {len(cls_score_list)} 0: {cls_score_list[0].shape} 1: {cls_score_list[1].shape} 2: {cls_score_list[2].shape}')
        # print(f'objectness_list: {len(objectness_list)} 0: {objectness_list[0].shape} 1: {objectness_list[1].shape} 2: {objectness_list[2].shape}')
        # return cls_score_list, reg_dist_list

        return outputs
