from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = (
            torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        )  # shift x
        sy = (
            torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        )  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2).transpose(0, 1))
        stride_tensor.append(
            torch.full((h * w, 1), stride, dtype=dtype, device=device).transpose(0, 1)
        )
    return anchor_points, stride_tensor
    # return torch.cat(anchor_points), torch.cat(stride_tensor)


class DetectV5(nn.Module):
    """YOLOv5 Detect head for detection models."""

    def __init__(self, old_detect):
        super().__init__()
        self.nc = old_detect.nc  # number of classes
        self.no = old_detect.no  # number of outputs per anchor
        self.nl = old_detect.nl  # number of detection layers
        self.na = old_detect.na
        self.grid = old_detect.grid  # [torch.zeros(1)] * self.nl
        self.anchor_grid = old_detect.anchor_grid
        self.m = old_detect.m
        self.inplace = old_detect.inplace
        self.stride = old_detect.stride
        self.anchors = old_detect.anchors
        self.f = old_detect.f
        self.i = old_detect.i

    def forward(self, x):
        outputs = []

        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            channel_output = torch.sigmoid(x[i])
            outputs.append(channel_output)

        return outputs


class DetectV7(nn.Module):
    """YOLOv7 Detect head for detection models."""

    def __init__(self, old_detect):
        super().__init__()
        self.nc = old_detect.nc  # number of classes
        self.no = old_detect.no  # number of outputs per anchor
        self.nl = old_detect.nl  # number of detection layers
        self.na = old_detect.na
        self.grid = old_detect.grid
        self.anchor_grid = old_detect.anchor_grid
        self.m = old_detect.m
        self.stride = old_detect.stride
        self.anchors = old_detect.anchors
        self.f = old_detect.f
        self.i = old_detect.i

    def forward(self, x):
        outputs = []

        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            channel_output = torch.sigmoid(x[i])
            outputs.append(channel_output)

        return outputs


class DetectV6R1(nn.Module):
    """Efficient Decoupled Head With hardware-aware degisn, the decoupled head is
    optimized with hybridchannels methods."""

    # def __init__(self, num_classes=80, anchors=1, num_layers=3, inplace=True, head_layers=None, use_dfl=True, reg_max=16):  # detection layer
    def __init__(self, old_detect):  # detection layer
        super().__init__()
        self.nc = old_detect.nc  # number of classes
        self.no = old_detect.no  # number of outputs per anchor
        self.nl = old_detect.nl  # number of detection layers
        self.na = old_detect.na
        self.anchors = old_detect.anchors
        self.grid = old_detect.grid  # [torch.zeros(1)] * self.nl
        self.prior_prob = 1e-2
        self.inplace = old_detect.inplace
        stride = [8, 16, 32]  # strides computed during build
        self.stride = torch.tensor(stride)

        # Init decouple head
        self.stems = old_detect.stems
        self.cls_convs = old_detect.cls_convs
        self.reg_convs = old_detect.reg_convs
        self.cls_preds = old_detect.cls_preds
        self.reg_preds = old_detect.reg_preds
        # New
        self.obj_preds = old_detect.obj_preds

    def forward(self, x):
        outputs = []
        for i in range(self.nl):
            x[i] = self.stems[i](x[i])
            cls_x = x[i]
            reg_x = x[i]
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)
            obj_output = self.obj_preds[i](reg_feat)
            y = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
            outputs.append(y)
        return outputs


class DetectV6R3(nn.Module):
    """Efficient Decoupled Head for YOLOv6 R2&R3 With hardware-aware degisn, the
    decoupled head is optimized with hybridchannels methods."""

    def __init__(self, old_detect, use_rvc2: bool):  # detection layer
        super().__init__()
        self.nc = old_detect.nc  # number of classes
        self.no = old_detect.no  # number of outputs per anchor
        self.nl = old_detect.nl  # number of detection layers
        if hasattr(old_detect, "anchors"):
            self.anchors = old_detect.anchors
        self.grid = old_detect.grid  # [torch.zeros(1)] * self.nl
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
        if hasattr(old_detect, "cls_preds"):
            self.cls_preds = old_detect.cls_preds
        elif hasattr(old_detect, "cls_preds_af"):
            self.cls_preds = old_detect.cls_preds_af
        if hasattr(old_detect, "reg_preds"):
            self.reg_preds = old_detect.reg_preds
        elif hasattr(old_detect, "reg_preds_af"):
            self.reg_preds = old_detect.reg_preds_af

        self.use_rvc2 = use_rvc2

    def forward(self, x):
        outputs = []
        for i in range(self.nl):
            b, _, h, w = x[i].shape
            x[i] = self.stems[i](x[i])
            cls_x = x[i]
            reg_x = x[i]
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)

            if self.use_dfl:
                reg_output = reg_output.reshape(
                    [-1, 4, self.reg_max + 1, h * w]
                ).permute(0, 2, 1, 3)
                reg_output = self.proj_conv(F.softmax(reg_output, dim=1))[:, 0]
                reg_output = reg_output.reshape([-1, 4, h, w])

            cls_output = torch.sigmoid(cls_output)
            # conf, _ = cls_output.max(1, keepdim=True)
            if self.use_rvc2:
                conf, _ = cls_output.max(1, keepdim=True)
            else:
                conf = torch.ones(
                    (cls_output.shape[0], 1, cls_output.shape[2], cls_output.shape[3]),
                    device=cls_output.device,
                )
            output = torch.cat([reg_output, conf, cls_output], axis=1)
            outputs.append(output)

        return outputs


class DetectV6R4s(nn.Module):
    """Efficient Decoupled Head for YOLOv6 R4 nano & small With hardware-aware design,
    the decoupled head is optimized with hybridchannels methods."""

    def __init__(self, old_detect, use_rvc2: bool):  # detection layer
        super().__init__()
        self.nc = old_detect.nc  # number of classes
        self.no = old_detect.no  # number of outputs per anchor
        self.nl = old_detect.nl  # number of detection layers
        if hasattr(old_detect, "anchors"):
            self.anchors = old_detect.anchors
        self.grid = old_detect.grid  # [torch.zeros(1)] * self.nl
        self.prior_prob = 1e-2
        self.inplace = old_detect.inplace
        self.stride = old_detect.stride
        if hasattr(old_detect, "use_dfl"):
            self.use_dfl = old_detect.use_dfl
            # print(old_detect.use_dfl)
        if hasattr(old_detect, "reg_max"):
            self.reg_max = old_detect.reg_max
        if hasattr(old_detect, "proj_conv"):
            self.proj_conv = old_detect.proj_conv
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        # Init decouple head
        self.stems = old_detect.stems
        self.cls_convs = old_detect.cls_convs
        self.reg_convs = old_detect.reg_convs
        if hasattr(old_detect, "cls_preds"):
            self.cls_preds = old_detect.cls_preds
        elif hasattr(old_detect, "cls_preds_af"):
            self.cls_preds = old_detect.cls_preds_af
        if hasattr(old_detect, "reg_preds"):
            self.reg_preds = old_detect.reg_preds
        elif hasattr(old_detect, "reg_preds_af"):
            self.reg_preds = old_detect.reg_preds_af

        self.use_rvc2 = use_rvc2

    def forward(self, x):
        outputs = []

        for i in range(self.nl):
            x[i] = self.stems[i](x[i])
            cls_x = x[i]
            reg_x = x[i]
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)

            cls_output = torch.sigmoid(cls_output)

            if self.use_rvc2:
                conf, _ = cls_output.max(1, keepdim=True)
            else:
                conf = torch.ones(
                    (cls_output.shape[0], 1, cls_output.shape[2], cls_output.shape[3]),
                    device=cls_output.device,
                )
            output = torch.cat([reg_output, conf, cls_output], axis=1)
            outputs.append(output)

        return outputs


class DetectV6R4m(nn.Module):
    """Efficient Decoupled Head for YOLOv6 R4 medium & large With hardware-aware design,
    the decoupled head is optimized with hybridchannels methods."""

    def __init__(self, old_detect, use_rvc2: bool):  # detection layer
        super().__init__()
        self.nc = old_detect.nc  # number of classes
        self.no = old_detect.no  # number of outputs per anchor
        self.nl = old_detect.nl  # number of detection layers
        if hasattr(old_detect, "anchors"):
            self.anchors = old_detect.anchors
        self.grid = old_detect.grid  # [torch.zeros(1)] * self.nl
        self.prior_prob = 1e-2
        self.inplace = old_detect.inplace
        self.stride = old_detect.stride
        self.use_dfl = old_detect.use_dfl
        # print(old_detect.use_dfl)
        self.reg_max = old_detect.reg_max
        self.proj_conv = old_detect.proj_conv
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        # Init decouple head
        self.stems = old_detect.stems
        self.cls_convs = old_detect.cls_convs
        self.reg_convs = old_detect.reg_convs
        if hasattr(old_detect, "cls_preds"):
            self.cls_preds = old_detect.cls_preds
        elif hasattr(old_detect, "cls_preds_af"):
            self.cls_preds = old_detect.cls_preds_af
        if hasattr(old_detect, "reg_preds"):
            self.reg_preds = old_detect.reg_preds
        elif hasattr(old_detect, "reg_preds_af"):
            self.reg_preds = old_detect.reg_preds_af

        self.use_rvc2 = use_rvc2

    def forward(self, x):
        outputs = []

        for i in range(self.nl):
            b, _, h, w = x[i].shape
            x[i] = self.stems[i](x[i])
            cls_x = x[i]
            reg_x = x[i]
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)

            if self.use_dfl:
                reg_output = reg_output.reshape(
                    [-1, 4, self.reg_max + 1, h * w]
                ).permute(0, 2, 1, 3)
                reg_output = self.proj_conv(F.softmax(reg_output, dim=1)).reshape(
                    [-1, 4, h, w]
                )

            cls_output = torch.sigmoid(cls_output)

            if self.use_rvc2:
                conf, _ = cls_output.max(1, keepdim=True)
            else:
                conf = torch.ones(
                    (cls_output.shape[0], 1, cls_output.shape[2], cls_output.shape[3]),
                    device=cls_output.device,
                )
            output = torch.cat([reg_output, conf, cls_output], axis=1)
            outputs.append(output)

        return outputs


class DetectV8(nn.Module):
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, old_detect, use_rvc2: bool):
        super().__init__()
        self.nc = old_detect.nc  # number of classes
        self.nl = old_detect.nl  # number of detection layers
        self.reg_max = (
            old_detect.reg_max
        )  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = old_detect.no  # number of outputs per anchor
        self.stride = old_detect.stride  # strides computed during build

        self.cv2 = old_detect.cv2
        self.cv3 = old_detect.cv3
        self.f = old_detect.f
        self.i = old_detect.i

        self.use_rvc2 = use_rvc2

        self.proj_conv = nn.Conv2d(old_detect.dfl.c1, 1, 1, bias=False).requires_grad_(
            False
        )
        x = torch.arange(old_detect.dfl.c1, dtype=torch.float)
        self.proj_conv.weight.data[:] = nn.Parameter(x.view(1, old_detect.dfl.c1, 1, 1))

    def forward(self, x):
        bs = x[0].shape[0]  # batch size

        outputs = []
        for i in range(self.nl):
            box = self.cv2[i](x[i])
            h, w = box.shape[2:]

            # ------------------------------
            # DFL PART
            box = box.view(bs, 4, self.reg_max, h * w).permute(0, 2, 1, 3)
            box = self.proj_conv(F.softmax(box, dim=1))[:, 0]
            box = box.reshape([bs, 4, h, w])
            # ------------------------------

            cls = self.cv3[i](x[i])
            cls_output = cls.sigmoid()
            if self.use_rvc2:
                conf, _ = cls_output.max(1, keepdim=True)
            else:
                conf = torch.ones(
                    (cls_output.shape[0], 1, cls_output.shape[2], cls_output.shape[3]),
                    device=cls_output.device,
                )

            output = torch.cat([box, conf, cls_output], axis=1)
            outputs.append(output)

        return outputs


class OBBV8(DetectV8):
    """YOLOv8 OBB detection head for detection with rotation models."""

    def __init__(self, old_obb, use_rvc2):
        super().__init__(old_obb, use_rvc2)
        self.ne = old_obb.ne  # number of extra parameters
        self.cv4 = old_obb.cv4

    def forward(self, x):
        # Detection part
        outputs = super().forward(x)

        # OBB part
        bs = x[0].shape[0]  # batch size
        angle = torch.cat(
            [self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
        )  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # Append the angle
        outputs.append(angle)

        return outputs


class PoseV8(DetectV8):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, old_kpts, use_rvc2):
        super().__init__(old_kpts, use_rvc2)
        self.kpt_shape = (
            old_kpts.kpt_shape
        )  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = old_kpts.nk  # number of keypoints total
        self.cv4 = old_kpts.cv4
        self.use_rvc2 = use_rvc2

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        if self.shape != bs:
            self.anchors, self.strides = make_anchors(x, self.stride, 0.5)
            self.shape = bs

        # Detection part
        outputs = super().forward(x)

        # Pose part
        for i in range(self.nl):
            kpt = self.cv4[i](x[i]).view(bs, self.nk, -1)
            outputs.append(self.kpts_decode(bs, kpt, i))

        return outputs

    def kpts_decode(self, bs, kpts, i):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        y = kpts.view(bs, *self.kpt_shape, -1)
        a = (y[:, :, :2] * 2.0 + (self.anchors[i] - 0.5)) * self.strides[i]
        if ndim == 3:
            # a = torch.cat((a, y[:, :, 2:3].sigmoid()*10), 2)
            a = torch.cat((a, y[:, :, 2:3]), 2)
        return a.view(bs, self.nk, -1)


class SegmentV8(DetectV8):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, old_segment, use_rvc2):
        super().__init__(old_segment, use_rvc2)
        self.nm = old_segment.nm  # number of masks
        self.npr = old_segment.npr  # number of protos
        self.proto = old_segment.proto  # protos
        self.cv4 = old_segment.cv4

    @staticmethod
    def _mask_call(layer, t):
        # Support both signatures: layer(t) and layer(t, t)
        try:
            return layer(t)
        except TypeError:
            return layer(t, t)

    def forward(self, x):
        # Detection part
        outputs = super().forward(x)
        # Masks
        outputs.extend(self._mask_call(self.cv4[i], x[i]) for i in range(self.nl))
        # Mask protos
        outputs.append(self.proto(x[0]))

        return outputs


class ClassifyV8(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, old_classify, use_rvc2: bool):
        super().__init__()
        self.conv = old_classify.conv
        self.pool = old_classify.pool
        self.drop = old_classify.drop
        self.linear = old_classify.linear
        self.f = old_classify.f
        self.i = old_classify.i

        self.use_rvc2 = use_rvc2

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x


class DetectV10(DetectV8):
    """YOLOv10 Detect head for detection models."""

    def __init__(self, old_detect, use_rvc2):
        super().__init__(old_detect, use_rvc2)
        self.cv2 = old_detect.one2one_cv2
        self.cv3 = old_detect.one2one_cv3


class DetectV26(nn.Module):
    """YOLOv26 Detect head for end-to-end NMS-free detection models.

    Uses one2one_cv2 and one2one_cv3 weights instead of cv2 and cv3 to enable NMS-free
    inference. The one2one heads are trained with tal_topk=1 for one-to-one label
    assignment.
    """

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    max_det = 300  # max detections per image

    def __init__(
        self,
        old_detect,
        use_rvc2: bool,
        conf_threshold: float = 0.0,
    ):
        super().__init__()
        self.nc = old_detect.nc  # number of classes
        self.nl = old_detect.nl  # number of detection layers
        self.reg_max = old_detect.reg_max  # DFL channels
        self.no = old_detect.no  # number of outputs per anchor
        self.stride = old_detect.stride  # strides computed during build

        # Use one2one heads for NMS-free inference
        self.cv2 = old_detect.one2one_cv2
        self.cv3 = old_detect.one2one_cv3
        self.f = old_detect.f
        self.i = old_detect.i

        self.use_rvc2 = use_rvc2
        self.conf_threshold = conf_threshold

    def forward(self, x):
        bs = x[0].shape[0]  # batch size

        boxes = []
        scores = []
        for i in range(self.nl):
            box = self.cv2[i](x[i])

            cls_regress = self.cv3[i](x[i])
            boxes.append(box.view(bs, 4, -1))
            scores.append(cls_regress.view(bs, self.nc, -1))

        preds = {
            "boxes": torch.cat(boxes, dim=2),
            "scores": torch.cat(scores, dim=2),
            "feats": x,
        }

        dbox = self._get_decode_boxes(preds)
        y = torch.cat((dbox, preds["scores"].sigmoid()), 1)  # (bs, 4+nc, num_anchors)
        y = y.permute(0, 2, 1)  # (bs, num_anchors, 4+nc)
        return y

    def _get_decode_boxes(self, preds):
        # Emulate ultralytics.nn.modules.head.Detect._get_decode_boxes for end2end export.
        # preds["boxes"]: (N, 4, A), preds["feats"]: list of feature maps (N, C, H_i, W_i)
        shape = preds["feats"][0].shape  # BCHW
        if self.dynamic or self.shape != shape:
            anchor_points, stride_tensor = self._make_anchors(
                preds["feats"], self.stride, 0.5
            )
            self.anchors = anchor_points.transpose(0, 1)
            self.strides = stride_tensor.transpose(0, 1)
            self.shape = shape

        # anchors: (1, 2, A), strides: (1, 1, A)
        # returns: decoded boxes (N, 4, A) in xyxy pixels
        dbox = self.dist2bbox(
            preds["boxes"], self.anchors.unsqueeze(0), xywh=False, dim=1
        )
        return dbox * self.strides

    @staticmethod
    def _make_anchors(feats, strides, grid_cell_offset=0.5):
        # Emulate ultralytics.utils.tal.make_anchors.
        # feats: list of (N, C, H_i, W_i) -> returns anchor_points (A, 2), stride_tensor (A, 1)
        anchor_points, stride_tensor = [], []
        dtype, device = feats[0].dtype, feats[0].device
        for i, stride in enumerate(strides):
            h, w = feats[i].shape[2:]
            sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
            sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
            sy, sx = torch.meshgrid(sy, sx, indexing="ij")
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(
                torch.full((h * w, 1), stride, dtype=dtype, device=device)
            )
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    @staticmethod
    def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
        # Emulate ultralytics.utils.tal.dist2bbox.
        # distance: (N, 4, A) if dim=1, anchor_points: (1, 2, A) -> returns (N, 4, A)
        # xywh=True outputs (cx, cy, w, h); xywh=False outputs (x1, y1, x2, y2)
        lt, rb = distance.chunk(2, dim)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return torch.cat([c_xy, wh], dim)
        return torch.cat((x1y1, x2y2), dim)


class SegmentV26(DetectV26):
    """YOLOv26 Segment head for end-to-end NMS-free instance segmentation models.

    Outputs decoded boxes, class scores, mask coefficients (separate), and prototype masks.

    Output format:
        - detections: (batch, num_anchors, 4 + nc)
            - 4: decoded bbox coordinates (x1, y1, x2, y2) in pixel space
            - nc: class scores (sigmoided)
        - mask_coeffs: (batch, num_anchors, nm)
            - nm: mask coefficients (raw, to be used with protos)
        - protos: (batch, nm, proto_h, proto_w)
            - Prototype masks

    To get final instance masks (K = number of kept detections):
        mask = sigmoid(mask_coefficients @ protos.flatten(2))  # (K, H*W)
        mask = mask.view(K, proto_h, proto_w)  # (K, proto_h, proto_w)
        mask = crop_to_bbox(mask, bbox)  # crop to detection bbox
    """

    def __init__(
        self,
        old_segment,
        use_rvc2: bool,
        conf_threshold: float = 0.0,
    ):
        super().__init__(old_segment, use_rvc2, conf_threshold)
        self.nm = old_segment.nm  # number of masks (default 32)
        self.npr = old_segment.npr  # number of protos (default 256)
        self.proto = old_segment.proto  # Proto26 module

        # Use one2one mask coefficient heads for NMS-free inference
        self.cv4 = old_segment.one2one_cv4

    def forward(self, x):
        """Forward pass returning detections, mask coefficients, and prototype masks.

        Args:
            x: List of feature maps from backbone [P3, P4, P5]

        Returns:
            Tuple of:
                - detections: (batch, num_anchors, 4 + nc)
                - mask_coeffs: (batch, num_anchors, nm)
                - protos: (batch, nm, proto_h, proto_w)
        """
        bs = x[0].shape[0]  # batch size

        boxes = []
        scores = []
        mask_coeffs = []
        for i in range(self.nl):
            # Box regression
            box = self.cv2[i](x[i])
            boxes.append(box.view(bs, 4, -1))

            # Class scores
            cls_regress = self.cv3[i](x[i])
            scores.append(cls_regress.view(bs, self.nc, -1))

            # Mask coefficients
            mask = self.cv4[i](x[i])
            mask_coeffs.append(mask.view(bs, self.nm, -1))

        preds = {
            "boxes": torch.cat(boxes, dim=2),
            "scores": torch.cat(scores, dim=2),
            "feats": x,
        }

        # Decode boxes to pixel coordinates
        dbox = self._get_decode_boxes(preds)

        # Detection output: boxes (4) + class scores (nc)
        y = torch.cat((dbox, preds["scores"].sigmoid()), 1)  # (bs, 4+nc, num_anchors)
        y = y.permute(0, 2, 1)  # (bs, num_anchors, 4+nc)

        # Mask coefficients output (separate)
        mask_coeffs_cat = torch.cat(mask_coeffs, dim=2)  # (bs, nm, num_anchors)
        mask_coeffs_cat = mask_coeffs_cat.permute(0, 2, 1)  # (bs, num_anchors, nm)

        # Get prototype masks
        proto = self._get_proto(x)

        return y, mask_coeffs_cat, proto

    def _get_proto(self, x):
        """Get prototype masks from Proto26 module.

        Emulate ultralytics.nn.modules.head.Segment26.forward for proto generation.

        Proto26 takes all feature maps and returns prototype masks.
        """
        return self.proto(x, return_semseg=False)


class PoseV26(DetectV26):
    """YOLOv26 Pose head for end-to-end NMS-free pose estimation models.

    Outputs decoded boxes, class scores, and decoded keypoints (separate).

    Output format:
        - detections: (batch, num_anchors, 4 + nc)
            - 4: decoded bbox coordinates (x1, y1, x2, y2) in pixel space
            - nc: class scores (sigmoided)
        - keypoints: (batch, num_anchors, nk)
            - nk: keypoint values (x, y, [visibility]) for each keypoint
            - x, y are in pixel coordinates
            - visibility is sigmoided (if ndim == 3)
    """

    def __init__(
        self,
        old_pose,
        use_rvc2: bool,
        conf_threshold: float = 0.0,
    ):
        super().__init__(old_pose, use_rvc2, conf_threshold)
        self.kpt_shape = old_pose.kpt_shape  # (num_keypoints, ndim) e.g., (17, 3)
        self.nk = old_pose.nk  # total keypoint values = num_keypoints * ndim

        # Pose26: cv4 is feature extractor, cv4_kpts produces keypoints
        self.cv4 = old_pose.one2one_cv4
        self.cv4_kpts = old_pose.one2one_cv4_kpts

    def forward(self, x):
        """Forward pass returning detections and decoded keypoints.

        Emulate ultralytics.nn.modules.head.Pose26.forward_head.

        Args:
            x: List of feature maps from backbone [P3, P4, P5]

        Returns:
            Tuple of:
                - detections: (batch, num_anchors, 4 + nc)
                - keypoints: (batch, num_anchors, nk)
        """
        bs = x[0].shape[0]  # batch size

        boxes = []
        scores = []
        kpts_raw = []
        for i in range(self.nl):
            # Box regression
            box = self.cv2[i](x[i])
            boxes.append(box.view(bs, 4, -1))

            # Class scores
            cls_regress = self.cv3[i](x[i])
            scores.append(cls_regress.view(bs, self.nc, -1))

            # Keypoints: cv4 extracts features, cv4_kpts predicts keypoints
            feat = self.cv4[i](x[i])
            kpt = self.cv4_kpts[i](feat)
            kpts_raw.append(kpt.view(bs, self.nk, -1))

        preds = {
            "boxes": torch.cat(boxes, dim=2),
            "scores": torch.cat(scores, dim=2),
            "feats": x,
        }

        # Decode boxes to pixel coordinates (this also sets self.anchors and self.strides)
        # from the parent DetectV26
        dbox = self._get_decode_boxes(preds)

        # Detection output: boxes (4) + class scores (nc)
        y = torch.cat((dbox, preds["scores"].sigmoid()), 1)  # (bs, 4+nc, num_anchors)
        y = y.permute(0, 2, 1)  # (bs, num_anchors, 4+nc)

        # Decode and concatenate keypoints
        # Note: After _get_decode_boxes, self.anchors is (2, A) and self.strides is (1, A)
        kpts_cat = torch.cat(kpts_raw, dim=2)  # (bs, nk, num_anchors)
        kpts_decoded = self._kpts_decode(bs, kpts_cat)  # (bs, nk, num_anchors)
        kpts_decoded = kpts_decoded.permute(0, 2, 1)  # (bs, num_anchors, nk)

        return y, kpts_decoded

    def _kpts_decode(self, bs, kpts):
        """Decode keypoints from raw predictions to pixel coordinates.

        Emulate ultralytics.nn.modules.head.Pose26.kpts_decode.

        Args:
            bs: Batch size
            kpts: Raw keypoint predictions (bs, nk, num_anchors)

        Returns:
            Decoded keypoints (bs, nk, num_anchors) with x, y in pixel coords
        """
        ndim = self.kpt_shape[1]
        num_kpts = self.kpt_shape[0]
        num_anchors = kpts.shape[2]

        # Reshape to (bs, num_keypoints, ndim, num_anchors)
        y = kpts.view(bs, num_kpts, ndim, num_anchors)

        # After _get_decode_boxes, anchors and strides are already in the right format:
        # self.anchors: (2, num_anchors), self.strides: (1, num_anchors)
        # Reshape for broadcasting with y[:, :, :2, :] which is (bs, num_kpts, 2, num_anchors)
        anchors_reshaped = self.anchors.view(1, 1, 2, num_anchors)  # (1, 1, 2, A)
        strides_reshaped = self.strides.view(1, 1, 1, num_anchors)  # (1, 1, 1, A)

        # Decode xy: (raw + anchor) * stride
        xy = (y[:, :, :2, :] + anchors_reshaped) * strides_reshaped

        if ndim == 3:
            # Visibility score (sigmoid)
            vis = y[:, :, 2:3, :].sigmoid()
            decoded = torch.cat((xy, vis), dim=2)
        else:
            decoded = xy

        # Reshape back to (bs, nk, num_anchors)
        return decoded.view(bs, self.nk, num_anchors)
