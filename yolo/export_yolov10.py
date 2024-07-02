import sys
sys.path.append("./yolo/ultralytics")

import re
import torch
import onnxsim
import onnx
from pathlib import Path
import json


from exporter import Exporter
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.nn.modules import Detect
from yolo.detect_head import DetectV10


DIR_TMP = "./tmp"

class YoloV10Exporter(Exporter):

    def __init__(self, conv_path, weights_filename, imgsz, conv_id, n_shaves=6, use_legacy_frontend='false', use_rvc2='true'):
        super().__init__(conv_path, weights_filename, imgsz, conv_id, n_shaves, use_legacy_frontend, use_rvc2)
        self.load_model()
    
    def load_model(self):
        # load the model
        # model = attempt_load_weights(str(self.weights_path.resolve()), device="cpu", inplace=True, fuse=True)
        model, _ = attempt_load_one_weight(str(self.weights_path.resolve()), device="cpu", inplace=True, fuse=True)

        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        
        # check num classes and labels
        # assert model.nc == len(names), f'Model class count {model.nc} != len(names) {len(names)}'
        assert model.yaml["nc"] == len(names), f'Model class count {model.yaml["nc"]} != len(names) {len(names)}'

        # Replace with the custom Detection Head
        if isinstance(model.model[-1], (Detect)):
            model.model[-1] = DetectV10(model.model[-1], self.use_rvc2)

        self.num_branches = model.model[-1].nl

        # check if image size is suitable
        gs = max(int(model.stride.max()), 32)  # model stride
        if isinstance(self.imgsz, int):
            self.imgsz = [self.imgsz, self.imgsz]
        for sz in self.imgsz:
            if sz % gs != 0:
                raise ValueError(f"Image size is not a multiple of maximum stride {gs}")

        # ensure correct length
        if len(self.imgsz) != 2:
            raise ValueError(f"Image size must be of length 1 or 2.")

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
                        output_names=[f"output{i+1}_yolov6r2" for i in range(self.num_branches)],
                        dynamic_axes=None)

        # check if the arhcitecture is correct
        model_onnx = onnx.load(self.f_onnx)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # simplify the moodel
        onnx_model, check = onnxsim.simplify(model_onnx)
        assert check, 'assert check failed'
    
        onnx.checker.check_model(onnx_model)  # check onnx model

        # save the simplified model
        self.f_simplified = (self.conv_path / f"{self.model_name}-simplified.onnx").resolve()
        onnx.save(onnx_model, self.f_simplified)
        return self.f_simplified
    
    def export_openvino(self, version):
        super().export_openvino('v6r2')

        if not self.use_rvc2:
            # Replace opset8 with opset1 for Softmax layers
            # Read the content of the file
            with open(self.f_xml, 'r') as file:
                content = file.read()

            # Use the re.sub() function to replace the pattern with the new version
            new_content = re.sub(r'type="SoftMax" version="opset8"', 'type="SoftMax" version="opset1"', content)

            # Write the updated content back to the file
            with open(self.f_xml, 'w') as file:
                file.write(new_content)

        return self.f_xml, self.f_mapping, self.f_bin
    
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
        content["nn_config"]["NN_specific_metadata"]["iou_threshold"] = 1.0
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
    
    def export_json(self):
        # generate anchors and sides
        anchors, masks = [], {}

        nc = self.model.model[-1].nc
        names = [f"Class_{i}" for i in range(nc)]

        return self.write_json(anchors, masks, nc, names)
    

if __name__ == "__main__":
    # Test the YoloV10Exporter
    conv_path = Path(DIR_TMP)
    weights_filename = "yolov10n.pt"
    imgsz = 640
    conv_id = "test"
    nShaves = 6
    useLegacyFrontend = 'false'
    useRVC2 = 'false'
    exporter = YoloV10Exporter(conv_path, weights_filename, imgsz, conv_id, nShaves, useLegacyFrontend, useRVC2)
    exporter.export_onnx()
    exporter.export_openvino("v6r2")
    exporter.export_json()
