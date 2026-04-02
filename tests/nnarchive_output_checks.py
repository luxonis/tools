from __future__ import annotations

from copy import deepcopy

V8_DETECTION_CHECK = {
    "name": "yolov8n",
    "version": "v8",
    "model_outputs": ["output1_yolov6r2", "output2_yolov6r2", "output3_yolov6r2"],
    "head_outputs": ["output1_yolov6r2", "output2_yolov6r2", "output3_yolov6r2"],
    "yolo_outputs": ["output1_yolov6r2", "output2_yolov6r2", "output3_yolov6r2"],
}

V8_SEG_CHECK = {
    "name": "yolov8n-seg",
    "version": "v8",
    "model_outputs": [
        "output1_yolov8",
        "output2_yolov8",
        "output3_yolov8",
        "output1_masks",
        "output2_masks",
        "output3_masks",
        "protos_output",
    ],
    "head_outputs": [
        "output1_yolov8",
        "output2_yolov8",
        "output3_yolov8",
        "output1_masks",
        "output2_masks",
        "output3_masks",
        "protos_output",
    ],
    "yolo_outputs": ["output1_yolov8", "output2_yolov8", "output3_yolov8"],
    "mask_outputs": ["output1_masks", "output2_masks", "output3_masks"],
}

V8_POSE_CHECK = {
    "name": "yolov8n-pose",
    "version": "v8",
    "model_outputs": [
        "output1_yolov8",
        "output2_yolov8",
        "output3_yolov8",
        "kpt_output1",
        "kpt_output2",
        "kpt_output3",
    ],
    "head_outputs": [
        "output1_yolov8",
        "output2_yolov8",
        "output3_yolov8",
        "kpt_output1",
        "kpt_output2",
        "kpt_output3",
    ],
    "yolo_outputs": ["output1_yolov8", "output2_yolov8", "output3_yolov8"],
    "keypoints_outputs": ["kpt_output1", "kpt_output2", "kpt_output3"],
}


def _clone_check(base_case: dict, *, name: str, version: str) -> dict:
    case = deepcopy(base_case)
    case["name"] = name
    case["version"] = version
    return case


N_VARIANT_OUTPUT_NAME_CHECKS = [
    V8_DETECTION_CHECK,
    V8_SEG_CHECK,
    V8_POSE_CHECK,
    _clone_check(V8_DETECTION_CHECK, name="yolov9t", version="v9"),
    _clone_check(V8_DETECTION_CHECK, name="yolov11n", version="v11"),
    _clone_check(V8_SEG_CHECK, name="yolov11n-seg", version="v11"),
    _clone_check(V8_POSE_CHECK, name="yolov11n-pose", version="v11"),
    _clone_check(V8_DETECTION_CHECK, name="yolov12n", version="v12"),
    {
        "name": "yolo26n",
        "version": "v26",
        "model_outputs": ["output_yolo26"],
        "head_outputs": ["output_yolo26"],
        "yolo_outputs": ["output_yolo26"],
    },
    {
        "name": "yolo26n-seg",
        "version": "v26",
        "model_outputs": ["output_yolo26", "output_masks", "protos_output"],
        "head_outputs": ["output_yolo26", "output_masks", "protos_output"],
        "yolo_outputs": ["output_yolo26"],
        "mask_outputs": ["output_masks"],
    },
    {
        "name": "yolo26n-pose",
        "version": "v26",
        "model_outputs": ["output_yolo26", "kpt_output"],
        "head_outputs": ["output_yolo26", "kpt_output"],
        "yolo_outputs": ["output_yolo26"],
        "keypoints_outputs": ["kpt_output"],
    },
]
