import sys
sys.path.append("./yolo/YOLOv6")

import torch
import json
import warnings
from yolov6.layers.common import RepVGGBlock
from yolov6.utils.checkpoint import load_checkpoint
import torch.nn as nn
import onnx
import onnxsim
import mo.main as model_optimizer
import subprocess
import blobconverter
import numpy as np
import openvino.inference_engine as ie
from zipfile import ZipFile
import os
from pathlib import Path

sys.path.remove('./yolo/YOLOv6')

DIR_TMP = "./tmp"

class YoloV6Exporter:

    def __init__(self, conv_path, weights_filename, imgsz, conv_id):

        # set up variables
        self.conv_path = conv_path
        self.weights_path = self.conv_path / weights_filename
        self.imgsz = imgsz
        self.model_name = "result"#weights_filename.split(".")[0]
        self.conv_id = conv_id

        # load the model
        self.load_model()

        # set up file paths
        self.f_onnx = None
        self.f_simplified = None
        self.f_bin = None
        self.f_xml = None
        self.f_mapping = None
        self.f_blob = None
        self.f_json = None
        self.f_zip = None

    
    def load_model(self):

        # code based on export.py from YoloV5 repository
        # load the model
        #model = DetectBackend(str(self.weights_path.resolve()), device="cpu")
        model = load_checkpoint(str(self.weights_path.resolve()), map_location="cpu", inplace=True, fuse=True)  # load FP32 model
        for layer in model.modules():
            #print(type(layer))
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
        
        # fuse RepVGG blocks
        #for layer in model.modules():
        #    print(type(layer))
        #    if isinstance(layer, RepVGGBlock):
        #        layer.switch_to_deploy()

        model.eval()
        self.model = model

    def export_onnx(self):
        # export onnx model
        self.f_onnx = (self.conv_path / f"{self.model_name}.onnx").resolve()
        im = torch.zeros(1, 3, *self.imgsz[::-1])#.to(device)  # image size(1,3,320,192) BCHW iDetection
        torch.onnx.export(self.model, im, self.f_onnx, verbose=False, opset_version=12,
                        training=torch.onnx.TrainingMode.EVAL,
                        do_constant_folding=True,
                        input_names=['images'],
                        output_names=['output'],
                        dynamic_axes=None)

        # check if the arhcitecture is correct
        model_onnx = onnx.load(self.f_onnx)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # simplify the moodel
        onnx_model, check = onnxsim.simplify(model_onnx)
        assert check, 'assert check failed'

        # get contacts ready for parsing
        conc_idx = []
        for i, n in enumerate(onnx_model.graph.node):
            if "Concat" in n.name:
                conc_idx.append(i)

        outputs = conc_idx[-(self.num_branches+1):-1]

        #output_names = []
        for i, idx in enumerate(outputs):
            onnx_model.graph.node[idx].name = f"output{i+1}_yolov6"
            #output_names.append(onnx_model.graph.node[idx].name)

        onnx.checker.check_model(onnx_model)  # check onnx model

        # save the simplified model
        self.f_simplified = (self.conv_path / f"{self.model_name}-simplified.onnx").resolve()
        onnx.save(onnx_model, self.f_simplified)
        return self.f_simplified

    def export_openvino(self):

        if self.f_simplified is None:
            self.export_onnx()
        
        output_list = [f"output{i+1}_yolov6" for i in range(self.num_branches)]
        output_list = ",".join(output_list)

        # export to OpenVINO and prune the model in the process
        cmd = f"mo --input_model '{self.f_simplified}' " \
        f"--output_dir '{self.conv_path.resolve()}' " \
        f"--model_name '{self.model_name}' " \
        '--data_type FP16 ' \
        '--reverse_input_channel ' \
        '--scale 255 ' \
        f'--output "{output_list}"'

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

    def export_blob(self):

        if self.f_xml is None or self.f_bin is None:
            self.export_openvino()
        
        # export blob from generate bin and xml
        blob_path = blobconverter.from_openvino(
            xml=str(self.f_xml.resolve()),#as_posix(),
            bin=str(self.f_bin.resolve()),#as_posix(),
            data_type="FP16",
            shaves=6,
            version="2021.4",
            use_cache=False,
            output_dir=self.conv_path.resolve()
        )

        self.f_blob = blob_path

        return blob_path

    def export_json(self):

        # load json template
        #f = open("./yolo/json/yolov5.json")
        f = open((Path(__file__).parent / "json" / "yolo.json").resolve())
        #print(f.resolve())
        content = json.load(f)

        # generate anchors and sides
        anchors, masks = [], {}

        nc = self.model.detect.nc

        # set parameters
        content["nn_config"]["input_size"] = "x".join([str(x) for x in self.imgsz])
        content["nn_config"]["NN_specific_metadata"]["classes"] = nc
        content["nn_config"]["NN_specific_metadata"]["anchors"] = anchors
        content["nn_config"]["NN_specific_metadata"]["anchor_masks"] = masks
        # use COCO labels if 80 classes, else use a placeholder
        content["mappings"]["labels"] = content["mappings"]["labels"] if nc == 80 else [f"Class_{i}" for i in range(nc)]

        # save json
        f_json = (self.conv_path / f"{self.model_name}.json").resolve()
        with open(f_json, 'w') as outfile:
            json.dump(content, outfile, ensure_ascii=False, indent=4)

        self.f_json = f_json

        return self.f_json

    def make_zip(self):
        # create a ZIP folder
        if self.f_simplified is None:
            self.export_onnx()
        
        if self.f_xml is None:
            self.export_openvino()

        if self.f_blob is None:
            self.export_blob()
        
        if self.f_json is None:
            self.export_json()

        #f_zip = f"{DIR_TMP}/{self.model_name}.zip"
        f_zip = (self.conv_path / f"{self.model_name}.zip").resolve()
        
        zip_obj = ZipFile(f_zip, 'w')
        zip_obj.write(self.f_simplified, self.f_simplified.name)
        zip_obj.write(self.f_xml, self.f_xml.name)
        zip_obj.write(self.f_bin, self.f_bin.name)
        zip_obj.write(self.f_blob, self.f_blob.name)
        zip_obj.write(self.f_json, self.f_json.name)
        zip_obj.close()

        self.f_zip = f_zip
        return f_zip
