from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple

from loguru import logger

from tools.modules import DetectV10, Exporter
from tools.utils import get_first_conv2d_in_channels
from tools.utils.constants import Encoding

current_dir = os.path.dirname(os.path.abspath(__file__))
yolo_path = os.path.join(current_dir, "ultralytics")
sys.path.append(yolo_path)

from ultralytics.nn.modules import Detect  # noqa: E402
from ultralytics.nn.tasks import load_checkpoint  # noqa: E402


class YoloV10Exporter(Exporter):
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
            subtype="yolov10",
            output_names=["output1_yolov10", "output2_yolov10", "output3_yolov10"],
        )
        self.load_model()

    def load_model(self):
        # load the model
        model, _ = load_checkpoint(
            self.model_path, device="cpu", inplace=True, fuse=True
        )

        if isinstance(model.model[-1], (Detect)):
            model.model[-1] = DetectV10(model.model[-1], self.use_rvc2)

        self.names = (
            model.module.names if hasattr(model, "module") else model.names
        )  # get class names
        # check num classes and labels
        assert model.yaml["nc"] == len(self.names), (
            f"Model class count {model.yaml['nc']} != len(names) {len(self.names)}"
        )

        try:
            self.number_of_channels = get_first_conv2d_in_channels(model)
            # print(f"Number of channels: {self.number_of_channels}")
        except Exception as e:
            logger.error(f"Error while getting number of channels: {e}")

        # check if image size is suitable
        gs = max(int(model.stride.max()), 32)  # model stride
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

    def export_nn_archive(
        self, class_names: Optional[List[str]] = None, encoding: Encoding = Encoding.RGB
    ):
        """Export the model to NN archive format.

        Args:
            class_list (Optional[List[str]], optional): List of class names. Defaults to None.
            encoding (Encoding): Color encoding used in the input model. Defaults to RGB.
        """
        names = list(self.model.names.values())

        if class_names is not None:
            assert len(class_names) == len(names), (
                f"Number of the given class names {len(class_names)} does not match number of classes {len(names)} provided in the model!"
            )
            names = class_names

        self.f_nn_archive = (self.output_folder / f"{self.model_name}.tar.xz").resolve()

        self.make_nn_archive(
            class_list=names, n_classes=self.model.model[-1].nc, encoding=encoding
        )
