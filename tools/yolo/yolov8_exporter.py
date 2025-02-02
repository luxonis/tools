from __future__ import annotations

import sys
from typing import List, Optional, Tuple

import torch
from luxonis_ml.nn_archive import ArchiveGenerator
from luxonis_ml.nn_archive.config_building_blocks import (
    DataType,
    Head,
    InputType,
)
from luxonis_ml.nn_archive.config_building_blocks.base_models.head_metadata import (
    HeadClassificationMetadata,
)

from tools.modules import (
    OBBV8,
    ClassifyV8,
    DetectV8,
    Exporter,
    Multiplier,
    PoseV8,
    SegmentV8,
)
from tools.utils import get_first_conv2d_in_channels

sys.path.append("./tools/yolo/ultralytics")
from ultralytics.nn.modules import OBB, Classify, Detect, Pose, Segment  # noqa: E402
from ultralytics.nn.tasks import attempt_load_one_weight  # noqa: E402

DETECT_MODE = 0
SEGMENT_MODE = 1
OBB_MODE = 2
CLASSIFY_MODE = 3
POSE_MODE = 4


def get_output_names(mode: int) -> List[str]:
    """
    Get the output names based on the mode.

    Args:
        mode (int): Mode of the model

    Returns:
        List[str]: List of output names
    """
    if mode == DETECT_MODE:
        return ["output1_yolov6r2", "output2_yolov6r2", "output3_yolov6r2"]
    elif mode == SEGMENT_MODE:
        return [
            "output1_yolov8",
            "output2_yolov8",
            "output3_yolov8",
            "output1_masks",
            "output2_masks",
            "output3_masks",
            "protos_output",
        ]
    elif mode == OBB_MODE:
        return ["output1_yolov8", "output2_yolov8", "output3_yolov8", "angle_output"]
    elif mode == POSE_MODE:
        return [
            "output1_yolov8",
            "output2_yolov8",
            "output3_yolov8",
            "kpt_output1",
            "kpt_output2",
            "kpt_output3",
        ]
    return ["output"]


class YoloV8Exporter(Exporter):
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
            subtype="yolov8",
            output_names=["output1_yolov8", "output2_yolov8", "output3_yolov8"],
        )
        self.load_model()

    def load_model(self):
        # load the model
        model, _ = attempt_load_one_weight(
            self.model_path, device="cpu", inplace=True, fuse=True
        )

        self.mode = -1
        if isinstance(model.model[-1], (Segment)):
            model.model[-1] = SegmentV8(model.model[-1], self.use_rvc2)
            self.mode = SEGMENT_MODE
            # self.export_stage2_multiplier()
        elif isinstance(model.model[-1], (OBB)):
            model.model[-1] = OBBV8(model.model[-1], self.use_rvc2)
            self.mode = OBB_MODE
        elif isinstance(model.model[-1], (Pose)):
            model.model[-1] = PoseV8(model.model[-1], self.use_rvc2)
            self.mode = POSE_MODE
        elif isinstance(model.model[-1], (Classify)):
            model.model[-1] = ClassifyV8(model.model[-1], self.use_rvc2)
            self.mode = CLASSIFY_MODE
        elif isinstance(model.model[-1], (Detect)):
            model.model[-1] = DetectV8(model.model[-1], self.use_rvc2)
            self.mode = DETECT_MODE

        if self.mode in [DETECT_MODE, SEGMENT_MODE, OBB_MODE, POSE_MODE]:
            self.names = (
                model.module.names if hasattr(model, "module") else model.names
            )  # get class names
            # check num classes and labels
            assert model.nc == len(
                self.names
            ), f"Model class count {model.nc} != len(names) {len(self.names)}"

        try:
            self.number_of_channels = get_first_conv2d_in_channels(model)
            # print(f"Number of channels: {self.number_of_channels}")
        except Exception as e:
            print(f"Error while getting number of channels: {e}")

        # Get output names
        self.all_output_names = get_output_names(self.mode)

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

    def export_stage2_multiplier(self):
        """Export the stage 2 multiplier to ONNX format."""
        stage2_w = self.imgsz[0] // 4
        stage2_h = self.imgsz[1] // 4
        self.stage2_filename = f"mult_{str(self.imgsz[0])}x{str(self.imgsz[1])}.onnx"
        self.f_stage2_onnx = (self.output_folder / self.stage2_filename).resolve()
        torch.onnx.export(
            Multiplier(),
            (torch.randn(1, 32, stage2_h, stage2_w), torch.randn(1, 32)),
            self.f_stage2_onnx,
            input_names=["prototypes", "coeffs"],
            output_names=["mask"],
        )

    def export_nn_archive(self, class_names: Optional[List[str]] = None):
        """
        Export the model to NN archive format.

        Args:
            class_list (Optional[List[str]], optional): List of class names. Defaults to None.
        """
        names = list(self.model.names.values())

        if class_names is not None:
            assert len(class_names) == len(
                names
            ), f"Number of the given class names {len(class_names)} does not match number of classes {len(names)} provided in the model!"
            names = class_names

        self.f_nn_archive = (self.output_folder / f"{self.model_name}.tar.xz").resolve()

        if self.mode == DETECT_MODE:
            self.make_nn_archive(names, self.model.model[-1].nc)
        elif self.mode == SEGMENT_MODE:
            self.make_nn_archive(
                names,
                self.model.model[-1].nc,
                parser="YOLOExtendedParser",
                # stage2_executable_path=str(self.f_stage2_onnx),
                # postprocessor_path=self.stage2_filename,
                n_prototypes=32,
                is_softmax=True,
                output_kwargs={
                    "mask_outputs": ["output1_masks", "output2_masks", "output3_masks"],
                    "protos_outputs": "protos_output",
                },
            )
        elif self.mode == OBB_MODE:
            self.make_nn_archive(
                names,
                self.model.model[-1].nc,
                output_kwargs={"angles_outputs": ["angle_output"]},
            )
        elif self.mode == POSE_MODE:
            self.make_nn_archive(
                names,
                self.model.model[-1].nc,
                parser="YOLOExtendedParser",
                n_keypoints=17,
                output_kwargs={"keypoints_outputs": ["kpt_output"]},
            )
        elif self.mode == CLASSIFY_MODE:
            self.make_cls_nn_archive(names, len(self.model.names))

    def make_cls_nn_archive(self, class_list: List[str], n_classes: int):
        """Export the model to NN archive format.

        Args:
            class_list (List[str], optional): List of class names
            n_classes (int): Number of classes
        """
        archive = ArchiveGenerator(
            archive_name=self.model_name,
            save_path=str(self.output_folder),
            cfg_dict={
                "config_version": "1.0",
                "model": {
                    "metadata": {
                        "name": self.model_name,
                        "path": f"{self.model_name}.onnx",
                    },
                    "inputs": [
                        {
                            "name": "images",
                            "dtype": DataType.FLOAT32,
                            "input_type": InputType.IMAGE,
                            "shape": [1, self.number_of_channels, *self.imgsz[::-1]],
                            "preprocessing": {
                                "mean": [0, 0, 0],
                                "scale": [255, 255, 255],
                            },
                        }
                    ],
                    "outputs": [
                        {
                            "name": output,
                            "dtype": DataType.FLOAT32,
                        }
                        for output in self.all_output_names
                    ],
                    "heads": [
                        Head(
                            parser="ClassificationParser",
                            metadata=HeadClassificationMetadata(
                                is_softmax=False,
                                n_classes=n_classes,
                                classes=class_list,
                            ),
                            outputs=self.all_output_names,
                        )
                    ],
                },
            },
            executables_paths=[str(self.f_onnx)],
        )
        archive.make_archive()
