# Tools-CLI

t

> \[!NOTE\]\
> This is the latest version of tools CLI. If you are looking for the tools web application, please refer to the [web-app](https://github.com/luxonis/tools/tree/web-app) branch.

This is a command-line tool that simplifies the conversion process of YOLO models. It supports the conversion of YOLOs ranging from V5 through V12 and Gold Yolo including oriented bounding boxes object detection (OBB), pose estimation, and instance segmentation variants of YOLOv8 and YOLO11 to ONNX format and archiving them in the NN Archive format.

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
PIP_CONSTRAINT=constraints.txt pip install .
# Running the package
tools yolov6nr4.pt --imgsz "416"
```

### Using Docker or Docker Compose

This option requires you to have Docker installed on your device. Additionally, to export a local model, please put it inside a `shared_with_container/models/` folder in the root folder of the project.

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

The output files are going to be in the `shared_with_container/outputs/` folder.

<a name="arguments"></a>

## ⚙️ Arguments

```
Usage: tools [OPTIONS] MODEL

Tools CLI

╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ --help (-h)  Display this message and exit.                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *  MODEL  Path or remote URI to the model file to convert. [required]        │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Parameters ─────────────────────────────────────────────────────────────────╮
│ --imgsz                   Input image size as either "width height" or a     │
│                           single "size" value applied to both dimensions.    │
│                           [default: 416 416]                                 │
│ --version                 YOLO variant to force, such as "yolov8". When      │
│                           omitted, the command runs automatic version        │
│                           detection. [default: None]                         │
│ --encoding                Color encoding used by the input model. Must be    │
│                           RGB or BGR. [choices: rgb, bgr] [default: rgb]     │
│ --use-rvc2 --no-use-rvc2  Whether to target RVC2 instead of RVC3. [default:  │
│                           True]                                              │
│ --class-names             Comma-separated class names recognized by the      │
│                           model. [default: None]                             │
│ --output-remote-url       Remote destination URL for uploading the generated │
│                           NN archive. [default: None]                        │
│ --put-file-plugin         Name of a function registered in PUT_FILE_REGISTRY │
│                           for uploads. [default: None]                       │
╰──────────────────────────────────────────────────────────────────────────────╯

```

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
| `yolov12`     | **Detection:** YOLO12n, YOLO12s, YOLO12m, YOLO12l, YOLO12x                                                                                                                                                                                                                          |
| `yolo26`      | **Detection, Instance Segmentation, Pose, Semantic Segmentation:** YOLO26n, YOLO26s, YOLO26m, YOLO26l, YOLO26x                                                                                                                                                                      |
| `yoloe`       | **Detection, Instance Segmentation:** YOLOE-11s, YOLOE-11m, YOLOE-11l; YOLOE-v8s, YOLOE-v8m, YOLOE-v8l                                                                                                                                                                              |
| `goldyolo`    | Gold-YOLO-N, Gold-YOLO-S, Gold-YOLO-M, Gold-YOLO-L                                                                                                                                                                                                                                  |

YOLOE models must be re-parameterized before conversion to be compatible with YOLOv8 and YOLOv11 formats.

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
