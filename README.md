# Tools-CLI

This application is used for exporting Yolo V5, V6, V7, V8 (OBB, instance segmentation, pose estimation, cls) and Gold YOLO object detection models to .ONNX.

## Running

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
docker run -v "$(PWD)/shared_with_container:/app/shared_with_container" tools-cli --model shared_with_container/yolov8n-seg.pt --imgsz "416"
```

### Using Docker compose

```bash
# Building Docker image
docker compose build
# Running the image
docker compose run tools-cli --model shared_with_container/yolov8n-seg.pt
```

### Using Python package

```bash
# Building the package
pip install .
# Running the package
tools --model shared_with_container/yolov8n-seg.pt --imgsz "416"
```

## Credits

This application uses source code of the following repositories: [YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6), [GoldYOLO](https://github.com/huawei-noah/Efficient-Computing) [YOLOv7](https://github.com/WongKinYiu/yolov7), and [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (see each of them for more information).

## License

This application is available under **AGPL-3.0 License** license (see [LICENSE](https://github.com/luxonis/tools/blob/master/LICENSE) file for details).
