# Tests of the Tools

This README describes the tests of the Tools app.

## Unit Tests

### Running

```
# To run the code with default arguments
python3 unittests.py
# To pass arguments you have to set them as environment variables specifying the output folders
export DOWNLOAD_WEIGHTS="True" && export DELETE_OUTPUT="True" && export tools_url="http://0.0.0.0" && export v5_folder="../../YoloV5Weights/" && export v6r1_folder="../../YOLOv6-Weights/R1/" && export v6r2_folder="../../YOLOv6-Weights/R2/" && export v6r21_folder="../../YOLOv6-Weights/R2.1/" && export v6r3_folder="../../YOLOv6-Weights/R3/" && export v6r4_folder="../../YOLOv6-Weights/R4/" && export v7_folder="../../YoloV7Weights/" && export v8_folder="../../YoloV8Weights/" && python3 unittests.py
# Downloading weights
export DOWNLOAD_WEIGHTS="True" && export DELETE_OUTPUT="True" && export tools_url="http://0.0.0.0" && python3 unittests.py
```

### List of supported models of the Unit tests

YoloV3

* yolov3-tinyu

YoloV5

* yolov5n
* yolov5s
* yolov5m
* yolov5l

YoloV6

* yolov6nr1
* yolov6tr1
* yolov6sr1
* yolov6nr2
* yolov6tr2
* yolov6sr2
* yolov6mr2
* yolov6nr21
* yolov6sr21
* yolov6mr21
* yolov6nr3
* yolov6sr3
* yolov6mr3
* yolov6nr4
* yolov6sr4
* yolov6mr4

YoloV7

* yolov7t
* yolov7

YoloV8

* yolov8n
* yolov8s
* yolov8m
* yolov8l