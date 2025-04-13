import sys
sys.path.append("./yolo/yolov5")
import torch

import yolov5.models.experimental
from yolov5.models.common import Conv
from yolov5.models.yolo import Detect
from yolov5.utils.activations import SiLU

import torch.nn as nn
import onnx
from exporter import Exporter
# import sparseml


def attempt_load(weights, device=None, inplace=True, fuse=True):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    from yolov5.models.yolo import Detect, Model  # noqa: E402

    model = yolov5.models.experimental.Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(
            yolov5.models.experimental.attempt_download(w),
            map_location="cpu",
            weights_only=False,
        )  # load
        ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, "stride"):
            ckpt.stride = torch.tensor([32.0])
        if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(
            ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval()
        )  # model in eval mode

    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace  # torch 1.7.0 compatibility
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")
                m.anchor_grid = [torch.zeros(1)] * m.nl
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))
    model.stride = model[
        torch.argmax(torch.tensor([m.stride.max() for m in model])).int()
    ].stride  # max stride
    assert all(
        model[0].nc == m.nc for m in model
    ), f"Models have different class counts: {[m.nc for m in model]}"
    return model


# Replace the original function
yolov5.models.experimental.attempt_load = attempt_load


class YoloV5Exporter(Exporter):
    def __init__(self, conv_path, weights_filename, imgsz, conv_id, n_shaves=6, use_legacy_frontend='false', use_rvc2='true'):
        super().__init__(conv_path, weights_filename, imgsz, conv_id, n_shaves, use_legacy_frontend, use_rvc2)
        self.load_model()

    def load_model(self):

        # code based on export.py from YoloV5 repository
        # load the model
        model = attempt_load(self.weights_path.resolve(), device=torch.device('cpu'))  # load FP32 model

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
        
        inplace = True

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
        onnx_model, check = self.get_onnx()
        assert check, 'assert check failed'

        # add named sigmoids for prunning in OpenVINO
        conv_indices = []
        for i, n in enumerate(onnx_model.graph.node):
            if "Conv" in n.name:
                conv_indices.append(i)

        inputs = conv_indices[-self.num_branches:]

        # Names of the node outputs you want to set as model outputs
        node_output_names = []
        for i, inp in enumerate(inputs):
            node_output_names.append(f'output{i+1}_yolov5')
            sigmoid = onnx.helper.make_node(
                'Sigmoid',
                inputs=[onnx_model.graph.node[inp].output[0]],
                outputs=[f'output{i+1}_yolov5'],
            )
            onnx_model.graph.node.append(sigmoid)

        # Create ValueInfoProto messages for the desired outputs
        output_value_infos = []
        for output_name in node_output_names:
            for node in onnx_model.graph.node:
                if output_name in node.output:
                    # Copy information from the output's ValueInfoProto
                    # (this includes the data type and shape, if they are known)
                    for value_info in onnx_model.graph.value_info:
                        if value_info.name == output_name:
                            output_value_infos.append(value_info)

        # Clear the existing outputs
        onnx_model.graph.ClearField('output')

        # Add the new outputs
        for value_info in output_value_infos:
            onnx_model.graph.output.add().CopyFrom(value_info)

        onnx.checker.check_model(onnx_model)  # check onnx model

        # save the simplified model
        self.f_simplified = (self.conv_path / f"{self.model_name}-simplified.onnx").resolve()
        onnx.save(onnx_model, self.f_simplified)
        return self.f_simplified

    def export_json(self):
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

        return self.write_json(anchors, masks)
