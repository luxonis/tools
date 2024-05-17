from __future__ import annotations

import os
from typing import List, Tuple

import onnx
import torch
from luxonis_ml.nn_archive import ArchiveGenerator
from luxonis_ml.nn_archive.config_building_blocks import HeadObjectDetectionYOLO
from luxonis_ml.nn_archive.config_building_blocks.base_models.head_outputs import (
    OutputsYOLO,
)

from tools.utils.constants import OUTPUTS_DIR


class Exporter:
    def __init__(
        self, model_path: str, imgsz: Tuple[int, int], use_rvc2: bool, subtype: str
    ):
        # Set up variables
        self.model_path = model_path
        self.imgsz = imgsz
        self.model_name = os.path.basename(self.model_path).split(".")[0]
        # Set up file paths
        self.f_onnx = None
        self.f_nn_archive = None
        self.output_names = None
        self.use_rvc2 = use_rvc2
        self.subtype = subtype

    def export_onnx(self, output_names: List[str] = ["output"]) -> os.PathLike:
        """Export the model to ONNX format.

        Args:
            output_names (List[str]): List of output names

        Returns:
            Path: Path to the exported ONNX model
        """
        # export onnx model
        self.f_onnx = (OUTPUTS_DIR / f"{self.model_name}.onnx").resolve()
        self.output_names = output_names
        im = torch.zeros(1, 3, *self.imgsz[::-1])
        torch.onnx.export(
            self.model,
            im,
            self.f_onnx,
            verbose=False,
            opset_version=12,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=["images"],
            output_names=output_names,
            dynamic_axes=None,
        )

        # Check if the arhcitecture is correct
        onnx.checker.check_model(self.f_onnx)
        return self.f_onnx

    def export_nn_archive(
        self,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.5,
        max_det: int = 300,
    ) -> os.PathLike:
        """Export the model to NN archive format.

        Args:
            iou_threshold (float): Intersection over Union threshold
            conf_threshold (float): Confidence threshold
            max_det (int): Maximum number of detections

        Returns:
            Path: Path to the exported NN archive
        """
        if self.f_onnx is None:
            raise ValueError("You need to export the ONNX model first")
        # export NN archive
        self.f_nn_archive = (OUTPUTS_DIR / self.model_name).resolve()

        class_list = (
            self.model.names
            if isinstance(self.model.names, list)
            else list(self.model.names.values())
        )

        archive = ArchiveGenerator(
            archive_name=self.model_name,
            save_path=str(self.f_nn_archive),
            cfg_dict={
                "config_version": "1.0",
                "model": {
                    "metadata": {
                        "name": self.model_name,
                        "path": self.f_onnx,
                    },
                    "inputs": [
                        {
                            "name": "images",
                            "dtype": "float32",
                            "input_type": "image",
                            "shape": [1, 3, *self.imgsz[::-1]],
                            "preprocessing": {
                                "mean": [0, 0, 0],
                                "scale": [255, 255, 255],
                                "reverse_channels": True,
                            },
                        }
                    ],
                    "outputs": [
                        {
                            "name": output,
                            "dtype": "float32",
                        }
                        for output in self.output_names
                    ],
                    "heads": [
                        HeadObjectDetectionYOLO(
                            family="ObjectDetectionYOLO",
                            outputs=OutputsYOLO(yolo_outputs=self.output_names),
                            n_classes=self.model.nc,
                            classes=class_list,
                            subtype=self.subtype,
                            iou_threshold=iou_threshold,
                            conf_threshold=conf_threshold,
                            max_det=max_det,
                        )
                    ],
                },
            },
            executables_paths=[self.f_onnx],
        )
        archive.make_archive()
        return self.f_nn_archive
