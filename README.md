# Tools-CLI

> \[!NOTE\]\
> This is the latest version of tools CLI. If you are looking for the tools web application, please refer to the [web-app](https://github.com/luxonis/tools/tree/web-app) branch.

This is a command-line tool that simplifies the conversion process of YOLO models. It supports the conversion of YOLOs ranging from V5 through V11 and Gold Yolo including oriented bounding boxes object detection (OBB), pose estimation, and instance segmentation variants of YOLOv8 and YOLO11 to ONNX format and archiving them in the NN Archive format.

> \[!WARNING\]\
> Please note that for the moment, we support conversion of YOLOv9 weights only from [Ultralytics](https://docs.ultralytics.com/models/yolov9/#performance-on-ms-coco-dataset).

## 📜 Table of contents

- [💻 How to run](#run)
- [⚙️ Arguments](#arguments)
- [🧰 Supported Models](#supported-models)
- [📝 Credits](#credits)
- [📄 License](#license)
- [🤝 Contributing](#contributing)

<a name="run"></a>

## 💻 How to run

You can either export a model stored on the cloud (e.g. S3) or locally. You can choose to install the toolkit through pip or using Docker. In the sections below, we'll describe both options.

### Prerequisites

```bash
# Cloning the tools repository and all submodules
git clone --recursive https://github.com/luxonis/tools.git
# Change folder
cd tools
```

### Using Python package

```bash
# Install the package 
pip install .
# Running the package 
tools yolov6nr4.pt --imgsz "416"
```

### Using Docker or Docker Compose

This option requires you to have Docker installed on your device. Additionally, to export a local model, please put it inside a `shared-component/models/` folder in the root folder of the project.

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

<a name="arguments"></a>

## ⚙️ Arguments

- `model: str` = Path to the model.
- `imgsz: str` = Image input shape in the format `width height` or `width`. Default value `"416 416"`.
- `version: Optional[str]` = Version of the YOLO model. Default value `None`. If not specified, the version will be detected automatically. Supported versions: `yolov5`, `yolov6r1`, `yolov6r3`, `yolov6r4`, `yolov7`, `yolov8`, `yolov9`, `yolov10`, `yolov11`, `goldyolo`.
- `use_rvc2: bool` = Whether to export for RVC2 or RVC3 devices. Default value `True`.
- `class_names: Optional[str]` = Optional list of classes separated by a comma, e.g. `"person, dog, cat"`
- `output_remote_url: Optional[str]` = Remote output url for the output .onnx model.
- `config_path: Optional[str]` = Optional path to an optional config.
- `put_file_plugin: Optional[str]` = Which plugin to use. Optional.

<a name="supported-models"></a>

## 🧰 Supported models

Currently, the following models are supported:

| Model Version | Supported versions                                                                                                                                                                                                                                                                  |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `yolov5`      | YOLOv5n, YOLOv5s, YOLOv5m, YOLOv5l, YOLOv5x, YOLOv5n6, YOLOv5s6, YOLOv5m6, YOLOv5l6                                                                                                                                                                                                 |
| `yolov6r1`    | **v1.0 release:** YOLOv6n, YOLOv6t, YOLOv6s                                                                                                                                                                                                                                         |
| `yolov6r3`    | **v2.0 release:** YOLOv6n, YOLOv6t, YOLOv6s, YOLOv6m, YOLOv6l <br/> **v2.1 release:** YOLOv6n, YOLOv6s, YOLOv6m, YOLOv6l <br/> **v3.0 release:** YOLOv6n, YOLOv6s, YOLOv6m, YOLOv6l                                                                                                 |
| `yolov6r4`    | **v4.0 release:** YOLOv6n, YOLOv6s, YOLOv6m, YOLOv6l                                                                                                                                                                                                                                |
| `yolov7`      | YOLOv7-tiny, YOLOv7, YOLOv7x                                                                                                                                                                                                                                                        |
| `yolov8`      | **Detection:** YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x, YOLOv3-tinyu, YOLOv5nu, YOLOv5n6u, YOLOv5s6u, YOLOv5su, YOLOv5m6u, YOLOv5mu, YOLOv5l6u, YOLOv5lu <br/> **Instance Segmentation, Pose, Oriented Detection, Classification:** YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x |
| `yolov9`      | YOLOv9t, YOLOv9s, YOLOv9m, YOLOv9c                                                                                                                                                                                                                                                  |
| `yolov10`     | YOLOv10n, YOLOv10s, YOLOv10m, YOLOv10b, YOLOv10l, YOLOv10x                                                                                                                                                                                                                          |
| `yolov11`     | **Detection, Instance Segmentation, Pose, Oriented Detection, Classification:** YOLO11n, YOLO11s, YOLO11m, YOLO11l, YOLO11x                                                                                                                                                         |
| `goldyolo`    | Gold-YOLO-N, Gold-YOLO-S, Gold-YOLO-M, Gold-YOLO-L                                                                                                                                                                                                                                  |

If you don't find your model in the list, it is possible that it can be converted, however, this is not guaranteed.

<a name="credits"></a>

## 📝 Credits

This application uses source code of the following repositories: [YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6), [GoldYOLO](https://github.com/huawei-noah/Efficient-Computing) [YOLOv7](https://github.com/WongKinYiu/yolov7), and [Ultralytics](https://github.com/ultralytics/ultralytics) (see each of them for more information).

<a name="license"></a>

## 📄 License

This application is available under **AGPL-3.0 License** license (see [LICENSE](https://github.com/luxonis/tools/blob/master/LICENSE) file for details).

<a name="contributing"></a>

## 🤝 Contributing

We welcome contributions! Whether it's reporting bugs, adding features or improving documentation, your help is much appreciated. Please create a pull request ([here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)'s how to do it) and assign anyone from the Luxonis team to review the suggested changes. Cheers!
