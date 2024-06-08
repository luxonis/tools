import sys

sys.path.append("./tools/yolov6r3/Efficient-Computing/Detection/Gold-YOLO/")
sys.path.append("./tools/yolov6r3/Efficient-Computing/Detection/Gold-YOLO/gold_yolo/")
sys.path.append("./tools/yolov6r3/Efficient-Computing/Detection/Gold-YOLO/yolov6/utils/")

from checkpoint import load_checkpoint as load_checkpoint_gold_yolo
from switch_tool import switch_to_deploy
from typing import Tuple

from tools.modules import Exporter, DetectV6R3
from tools.utils import get_first_conv2d_in_channels


class GoldYoloExporter(Exporter):
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
            subtype="yolov6",
            output_names=["output1_yolov6r2", "output2_yolov6r2", "output3_yolov6r2"],
        )
        self.load_model()

    def load_model(self):
        # Load the model
        model = load_checkpoint_gold_yolo(self.model_path, map_location="cpu")

        model.detect = DetectV6R3(model.detect, self.use_rvc2)
        self.num_branches = len(model.detect.grid)

        # switch to deploy
        model = switch_to_deploy(model)

        try:
            self.number_of_channels = get_first_conv2d_in_channels(model)
            # print(f"Number of channels: {self.number_of_channels}")
        except Exception as e:
            print(f"Error while getting number of channels: {e}")

        # check if image size is suitable
        gs = 2 ** (2 + self.num_branches)  # 1 = 8, 2 = 16, 3 = 32
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
