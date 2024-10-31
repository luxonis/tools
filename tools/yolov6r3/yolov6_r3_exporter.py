from __future__ import annotations

import sys
from typing import Tuple

from tools.modules import DetectV6R3, Exporter, YoloV6BackBone
from tools.utils import get_first_conv2d_in_channels

sys.path.append("tools/yolov6r3/YOLOv6R3")  # noqa: E402
from yolov6.layers.common import RepVGGBlock  # noqa: E402
from yolov6.utils.checkpoint import load_checkpoint  # noqa: E402

try:
    from yolov6.models.efficientrep import (
        CSPBepBackbone,  # noqa: E402
        CSPBepBackbone_P6,  # noqa: E402
        EfficientRep,  # noqa: E402
        EfficientRep6,  # noqa: E402
    )
except Exception as e:
    raise ImportError(
        "Error while importing EfficientRep, CSPBepBackbone, CSPBepBackbone_P6 or EfficientRep6: {e}"
    ) from e


class YoloV6R3Exporter(Exporter):
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
            output_names=["output1_yolov6r2", "output2_yolov6r2", "output3_yolov6r2"],
        )
        self.load_model()

    def load_model(self):
        # Code based on export.py from YoloV5 repository
        # load the model
        model = load_checkpoint(
            self.model_path,
            map_location="cpu",
            inplace=True,
            fuse=True,
        )  # load FP32 model

        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()

        for n, module in model.named_children():
            if isinstance(module, EfficientRep) or isinstance(module, CSPBepBackbone):
                setattr(model, n, YoloV6BackBone(module))
            elif isinstance(module, EfficientRep6):
                setattr(model, n, YoloV6BackBone(module, uses_6_erblock=True))
            elif isinstance(module, CSPBepBackbone_P6):
                setattr(
                    model,
                    n,
                    YoloV6BackBone(module, uses_fuse_P2=False, uses_6_erblock=True),
                )

        if not hasattr(model.detect, "obj_preds"):
            model.detect = DetectV6R3(model.detect, self.use_rvc2)

        self.num_branches = len(model.detect.grid)

        try:
            self.number_of_channels = get_first_conv2d_in_channels(model)
            # print(f"Number of channels: {self.number_of_channels}")
        except Exception as e:
            print(f"Error while getting number of channels: {e}")

        # check if image size is suitable
        gs = 2 ** (2 + self.num_branches)  # 1 = 8, 2 = 16, 3 = 32
        if isinstance(self.imgsz, int):
            self.imgsz = [self.imgsz, self.imgsz]
        for sz in self.imgsz:
            if sz % gs != 0:
                raise ValueError(f"Image size is not a multiple of maximum stride {gs}")

        # ensure correct length
        if len(self.imgsz) != 2:
            raise ValueError("Image size must be of length 1 or 2.")

        model.eval()
        self.model = model
