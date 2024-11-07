# Tools-CLI

> \[!NOTE\]\
> This is the latest version of tools CLI. If you are looking for the tools web application, please refer to the [web-app](https://github.com/luxonis/tools/tree/web-app) branch.

This is a command-line tool that simplifies the conversion process of YOLO models. It supports the conversion of YOLOs ranging from V5 through V11 and Gold Yolo including oriented bounding boxes object detection (OBB), pose estimation, and instance segmentation variants of YOLOv8 and YOLO11 to ONNX format.

> \[!NOTE\]\
> Please note that for the moment, we support conversion of YOLOv9 weights only from [Ultralytics](https://docs.ultralytics.com/models/yolov9/#performance-on-ms-coco-dataset).

## Running

You can either export a model stored on the cloud (e.g. S3) or locally. You can choose to install the toolkit through pip or using Docker. In the sections below, we'll describe both options.

### Using Python package

```bash
# Install the package
pip install tools@git+https://github.com/luxonis/tools.git@main
# Running the package
tools yolov6nr4.pt --imgsz "416"
```

### Using Docker or Docker Compose

This option requires you to have Docker installed on your device. Additionally, to export a local model, please put it inside a `shared-component/models/` folder in the root folder of the project.

#### Prerequisites

```bash
# Cloning the tools repository and all submodules
git clone --recursive https://github.com/luxonis/tools.git
# Change folder
cd tools
```

#### Using Docker

```bash
# Building Docker image
docker build -t tools_cli .
# Running the image
docker run -v "${PWD}/shared_with_container:/app/shared_with_container" tools_cli shared_with_container/models/yolov8n-seg.pt --imgsz "416"
```

#### Using Docker compose

```bash
# Building Docker image
docker compose build
# Running the image
docker compose run tools_cli shared_with_container/models/yolov6nr4.pt
```

The output files are going to be in `shared-component/output` folder.

### Arguments

- `model: str` = Path to the model.
- `imgsz: str` = Image input shape in the format `width height` or `width`. Default value `"416 416"`.
- `version: Optional[str]` =
- `use_rvc2: bool` = Whether to export for RVC2 or RVC3 devices. Default value `True`.
- `class_names: Optional[str]` = Optional list of classes separated by a comma, e.g. `"person, dog, cat"`
- `output_remote_url: Optional[str]` = Remote output url for the output .onnx model.
- `config_path: Optional[str]` = Optional path to an optional config.
- `put_file_plugin: Optional[str]` = Which plugin to use. Optional.

## Credits

This application uses source code of the following repositories: [YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6), [GoldYOLO](https://github.com/huawei-noah/Efficient-Computing) [YOLOv7](https://github.com/WongKinYiu/yolov7), and [Ultralytics](https://github.com/ultralytics/ultralytics) (see each of them for more information).

## License

This application is available under **AGPL-3.0 License** license (see [LICENSE](https://github.com/luxonis/tools/blob/master/LICENSE) file for details).
