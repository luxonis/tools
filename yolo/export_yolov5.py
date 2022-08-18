import sys
sys.path.append("./yolo/yolov5")

from pathlib import Path
import importlib

import torch
import json
import warnings

from yolov5.models.experimental import attempt_load
from yolov5.models.common import Conv
from yolov5.models.yolo import Detect

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

sys.path.remove("/app/yolo/yolov5")
sys.path.remove("./yolo/yolov5")

DIR_TMP = "./tmp"

class YoloV5Exporter:

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
        model = attempt_load(self.weights_path.resolve(), map_location=torch.device('cpu'))  # load FP32 model

        # check num classes and labels
        assert model.nc == len(model.names), f'Model class count {model.nc} != len(names) {len(model.names)}'

        # check if image size is suitable
        gs = int(max(model.stride))  # grid size (max stride)
        if isinstance(self.imgsz, int):
            self.imgsz = [self.imgsz, self.imgsz]
        for sz in self.imgsz:
            if sz % gs != 0:
                raise ValueError(f"Image size is not a multiple of maximum stride {gs}")

        # ensure correct length
        if len(self.imgsz) != 2:
            raise ValueError(f"Image size must be of length 1 or 2.")
        
        model.eval()
        for k, m in model.named_modules():
            if isinstance(m, Conv):  # assign export-friendly activations
                if isinstance(m.act, nn.SiLU):
                    m.act = SiLU()
            elif isinstance(m, Detect):
                m.inplace = inplace
                m.onnx_dynamic = False
                if hasattr(m, 'forward_export'):
                    m.forward = m.forward_export  # assign custom forward (optional)

        self.model = model

        m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
        self.num_branches = len(m.anchor_grid)           

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

        # add named sigmoids for prunning in OpenVINO
        conv_indices = []
        for i, n in enumerate(onnx_model.graph.node):
            if "Conv" in n.name:
                conv_indices.append(i)

        inputs = conv_indices[-self.num_branches:]

        for i, inp in enumerate(inputs):
            sigmoid = onnx.helper.make_node(
                'Sigmoid',
                inputs=[onnx_model.graph.node[inp].output[0]],
                outputs=[f'output{i+1}_yolov5'],
            )
            onnx_model.graph.node.append(sigmoid)

        onnx.checker.check_model(onnx_model)  # check onnx model

        # save the simplified model
        self.f_simplified = (self.conv_path / f"{self.model_name}-simplified.onnx").resolve()
        onnx.save(onnx_model, self.f_simplified)
        return self.f_simplified

    def export_openvino(self):

        if self.f_simplified is None:
            self.export_onnx()
        
        output_list = [f"output{i+1}_yolov5" for i in range(self.num_branches)]
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
        anchors, sides = [], []
        m = self.model.module.model[-1] if hasattr(self.model, 'module') else self.model.model[-1]
        for i in range(self.num_branches):
            sides.append(m.anchor_grid[i].size()[3])
            for j in range(m.anchor_grid[i].size()[1]):
                anchors.extend(m.anchor_grid[i][0, j, 0, 0].numpy())
        anchors = [float(x) for x in anchors]
        #sides.sort()

        # generate masks
        masks = dict()
        #for i, num in enumerate(sides[::-1]):
        for i, num in enumerate(sides):
            masks[f"side{num}"] = list(range(i*3, i*3+3))

        # set parameters
        content["nn_config"]["input_size"] = "x".join([str(x) for x in self.imgsz])
        content["nn_config"]["NN_specific_metadata"]["classes"] = self.model.nc
        content["nn_config"]["NN_specific_metadata"]["anchors"] = anchors
        content["nn_config"]["NN_specific_metadata"]["anchor_masks"] = masks
        content["mappings"]["labels"] = self.model.names

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
