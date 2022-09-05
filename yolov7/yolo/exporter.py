import json
import torch
import onnx
import onnxsim
import subprocess
import blobconverter
from zipfile import ZipFile
from pathlib import Path

class Exporter:
    def __init__(self, conv_path, weights_filename, imgsz, conv_id):

        # set up variables
        self.conv_path = conv_path
        self.weights_path = self.conv_path / weights_filename
        self.imgsz = imgsz
        self.model_name = weights_filename.split(".")[0] #"result"
        self.conv_id = conv_id

        # set up file paths
        self.f_onnx = None
        self.f_simplified = None
        self.f_bin = None
        self.f_xml = None
        self.f_mapping = None
        self.f_blob = None
        self.f_json = None
        self.f_zip = None

    def get_onnx(self):
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
        return onnxsim.simplify(model_onnx)


    def export_openvino(self, version):

        if self.f_simplified is None:
            self.export_onnx()
        
        output_list = [f"output{i+1}_yolo{version}" for i in range(self.num_branches)]
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

    def write_json(self, anchors, masks, nc = None, names = None):
        # set parameters
        f = open((Path(__file__).parent / "json" / "yolo.json").resolve())
        content = json.load(f)

        content["model"]["xml"] = f"{self.model_name}.xml"
        content["model"]["bin"] = f"{self.model_name}.bin"
        content["nn_config"]["input_size"] = "x".join([str(x) for x in self.imgsz])
        if nc:
            content["nn_config"]["NN_specific_metadata"]["classes"] = nc
        else:
            content["nn_config"]["NN_specific_metadata"]["classes"] = self.model.nc
        content["nn_config"]["NN_specific_metadata"]["anchors"] = anchors
        content["nn_config"]["NN_specific_metadata"]["anchor_masks"] = masks
        if names:
            # use COCO labels if 80 classes, else use a placeholder
            content["mappings"]["labels"] = content["mappings"]["labels"] if nc == 80 else names
        else:
            content["mappings"]["labels"] = self.model.names if isinstance(self.model.names, list) else list(self.model.names.values())
        content["version"] = 1

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
