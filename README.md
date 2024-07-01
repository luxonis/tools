# Tools-CLI

This application is used for exporting Yolo V5, V6, V7, V8 (OBB, instance segmentation, pose estimation, cls) and Gold YOLO object detection models to .ONNX.

## Running

You can either export a model stored on cloud (e.g. S3) or locally. To export a local model, please put it inside a `shared-component` folder.

The output files are going to be in `shared-component/output` folder.

### Prerequisites

```bash
# Cloning the tools repository and all submodules
git clone --recursive https://github.com/luxonis/tools.git
# Change folder
cd tools
```

### Using Docker

```bash
# Building Docker image
docker build -t tools-cli .
# Running the image
docker run -v "${PWD}/shared_with_container:/app/shared_with_container" tools-cli shared_with_container/models/yolov8n-seg.pt --imgsz "416"
```

### Using Docker compose

```bash
# Building Docker image
docker compose build
# Running the image
docker compose run tools-cli shared_with_container/models/yolov6nr4.pt
```

### Using Python package

```bash
# Building the package
pip install .
# Running the package
tools shared_with_container/models/yolov6nr4.pt --imgsz "416"
```

### Arguments

* `model: str` = Path to the model.
* `imgsz: str` = Image input shape in the format `width height` or `width`. Default value `"416 416"`.
* `version: Optional[str]` = 
* `use_rvc2: bool` = Whether to export for RVC2 or RVC3 devices. Default value `True`.
* `class_names: Optional[str]` = Optional list of classes separated by a comma, e.g. `"person, dog, cat"`
* `output_remote_url: Optional[str]` = Remote output url for the output .onnx model.
* `config_path: Optional[str]` = Optional path to an optional config.
* `put_file_plugin: Optional[str]` = Which plugin to use. Optional.

## Credits

This application uses source code of the following repositories: [YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6), [GoldYOLO](https://github.com/huawei-noah/Efficient-Computing) [YOLOv7](https://github.com/WongKinYiu/yolov7), and [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (see each of them for more information).

## License

This application is available under **AGPL-3.0 License** license (see [LICENSE](https://github.com/luxonis/tools/blob/master/LICENSE) file for details).
