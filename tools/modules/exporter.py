from __future__ import annotations

from datetime import datetime
import os
from typing import List, Tuple, Optional

import onnx
import onnxsim
import torch
from luxonis_ml.nn_archive import ArchiveGenerator
from luxonis_ml.nn_archive.config_building_blocks import (
    Head,
    InputType,
    DataType,
)
from luxonis_ml.nn_archive.config_building_blocks.base_models.head_metadata import (
    HeadYOLOMetadata,
)

from tools.utils.constants import OUTPUTS_DIR


class Exporter:
    """Exporter class to export models to ONNX and NN archive formats."""
    def __init__(
        self, 
        model_path: str, 
        imgsz: Tuple[int, int], 
        use_rvc2: bool, 
        subtype: str,
        output_names: List[str] = ["output"],
        all_output_names: Optional[List[str]] = None,
    ):
        """
        Initialize the Exporter class.

        Args:
            model_path (str): Path to the model's weights
            imgsz (Tuple[int, int]): Image size [width, height]
            use_rvc2 (bool): Whether to use RVC2
            subtype (str): Subtype of the model
            output_names (List[str]): List of output names. Defaults to ["output"].
            all_output_names (Optional[List[str]]): List of all output names. Defaults to None.
        """
        # Set up variables
        self.model_path = model_path
        self.imgsz = imgsz
        self.model_name = os.path.basename(self.model_path).split(".")[0]
        # Set up file paths
        self.f_onnx = None
        self.f_nn_archive = None
        self.use_rvc2 = use_rvc2
        self.number_of_channels = None
        self.subtype = subtype
        self.output_names = output_names
        self.all_output_names = all_output_names if all_output_names is not None else output_names
        self.output_folder = (OUTPUTS_DIR / f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
        # If output directory does not exist, create it
        if not self.output_folder.exists():
            self.output_folder.mkdir(parents=True)

    def export_onnx(self) -> os.PathLike:
        """Export the model to ONNX format.

        Returns:
            Path: Path to the exported ONNX model
        """
        self.f_onnx = (self.output_folder / f"{self.model_name}.onnx").resolve()
        im = torch.zeros(1, self.number_of_channels, *self.imgsz[::-1])
        # export onnx model
        torch.onnx.export(
            self.model,
            im,
            self.f_onnx,
            verbose=False,
            opset_version=12,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=["images"],
            output_names=self.all_output_names,
            dynamic_axes=None,
        )

        # check if the arhcitecture is correct
        model_onnx = onnx.load(self.f_onnx)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # simplify the moodel
        onnx_model, check = onnxsim.simplify(model_onnx)
        assert check, "Simplified ONNX model could not be validated"

        # Save onnx model
        onnx.save(onnx_model, self.f_onnx)

        return self.f_onnx

    def make_nn_archive(
        self,
        class_list: List[str],
        n_classes: int,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.5,
        max_det: int = 300,
        parser: str = "YOLO",
        stage2_executable_path: Optional[str] = None,
        postprocessor_path: Optional[str] = None,
        n_prototypes: Optional[int] = None,
        n_keypoints: Optional[int] = None,
        is_softmax: Optional[bool] = None,
        output_kwargs: Optional[dict] = {},
    ):
        """Export the model to NN archive format.

        Args:
            class_list (List[str], optional): List of class names
            n_classes (int): Number of classes
            iou_threshold (float): Intersection over Union threshold
            conf_threshold (float): Confidence threshold
            max_det (int): Maximum number of detections
            parser (str): Parser type, defaults to "YOLO"
            2stage_executable_path (Optional[str], optional): Path to the executables. Defaults to None.
            postprocessor_path (Optional[str], optional): Path to the postprocessor. Defaults to None.
            n_prototypes (Optional[int], optional): Number of prototypes. Defaults to None.
            n_keypoints (Optional[int], optional): Number of keypoints. Defaults to None.
            is_softmax (Optional[bool], optional): Whether to use softmax. Defaults to None.
            output_kwargs (Optional[dict], optional): Output keyword arguments. Defaults to None.
        """
        self.f_nn_archive = (self.output_folder / f"{self.model_name}.tar.xz").resolve()
        if stage2_executable_path is not None:
            executables_paths = [str(self.f_onnx), stage2_executable_path]
        else:
            executables_paths = [str(self.f_onnx)]
        
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
                                "reverse_channels": True,
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
                            parser=parser,
                            metadata=HeadYOLOMetadata(
                                yolo_outputs=self.output_names, 
                                subtype=self.subtype,
                                n_classes=n_classes,
                                classes=class_list,
                                iou_threshold=iou_threshold,
                                conf_threshold=conf_threshold,
                                max_det=max_det,
                                postprocessor_path=postprocessor_path,
                                n_prototypes=n_prototypes,
                                n_keypoints=n_keypoints,
                                is_softmax=is_softmax,
                                **output_kwargs,
                            ),
                            outputs=self.all_output_names,
                        )
                    ],
                },
            },
            executables_paths=executables_paths,
        )
        archive.make_archive()

    def export_nn_archive(self, class_names: Optional[List[str]] = None):
        """
        Export the model to NN archive format.
        
        Args:
            class_list (Optional[List[str]], optional): List of class names. Defaults to None.
        """
        nc = self.model.detect.nc
        # If class names are provided, use them
        if class_names is not None:
            assert len(class_names) == nc, f"Number of the given class names {len(class_names)} does not match number of classes {nc} provided in the model!"
            names = class_names
        else:
            # Check if the model has a names attribute
            if hasattr(self.model, "names"):
                names = self.model.names
            else:
                names = [f"Class_{i}" for i in range(nc)]

        self.make_nn_archive(names, nc)