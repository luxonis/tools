import sys

sys.path.append("tools/yolov6r3/YOLOv6R3")

from yolov6.layers.common import RepVGGBlock
from yolov6.models.efficientrep import (
    CSPBepBackbone,
    CSPBepBackbone_P6,
    EfficientRep,
    EfficientRep6,
)
from yolov6.utils.checkpoint import load_checkpoint
from typing import Tuple

from tools.modules import Exporter, YoloV6BackBone, DetectV6R3


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
