from __future__ import annotations

import sys
from typing import List, Optional, Tuple

from tools.modules import DetectV7, Exporter
from tools.utils import get_first_conv2d_in_channels

sys.path.append("./tools/yolov7/yolov7")
from models.experimental import attempt_load  # noqa: E402


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
        assert model.nc == len(
            model.names
        ), f"Model class count {model.nc} != len(names) {len(model.names)}"

        if hasattr(model, "module"):
            model.module.model[-1] = DetectV7(model.module.model[-1])
            # self.number_of_channels = model.module.model[0].conv.in_channels
        else:
            model.model[-1] = DetectV7(model.model[-1])
            # self.number_of_channels = model.model[0].conv.in_channels

        try:
            self.number_of_channels = get_first_conv2d_in_channels(model)
            # print(f"Number of channels: {self.number_of_channels}")
        except Exception as e:
            print(f"Error while getting number of channels: {e}")

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

        model.eval()

        self.model = model

        self.m = model.module.model[-1] if hasattr(model, "module") else model.model[-1]
        self.num_branches = len(self.m.anchor_grid)

    def export_nn_archive(self, class_names: Optional[List[str]] = None):
        """
        Export the model to NN archive format.

        Args:
            class_list (Optional[List[str]], optional): List of class names. Defaults to None.
        """
        names = self.model.names

        if class_names is not None:
            assert (
                len(class_names) == self.model.nc
            ), f"Number of the given class names {len(class_names)} does not match number of classes {self.model.nc} provided in the model!"
            names = class_names

        anchors = self.m.anchor_grid[:, 0, :, 0, 0].numpy().tolist()
        self.make_nn_archive(
            names, self.model.nc, parser="YOLOExtendedParser", anchors=anchors
        )
