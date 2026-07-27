from __future__ import annotations

import zipfile

import pytest
import torch

from tools.utils.constants import Encoding
from tools.version_detection import YOLOX_CONVERSION, detect_version
from tools.yolox.yolox_exporter import YoloXExporter, _infer_standard_architecture


def test_detects_yolox_state_dict_checkpoint(tmp_path) -> None:
    """Recognize YOLOX's state-dict markers without relying on its extension."""
    checkpoint = tmp_path / "weights.pth"
    markers = (
        "backbone.backbone.stem.conv.conv backbone.lateral_conv0 "
        "head.stems head.cls_convs head.reg_convs head.cls_preds "
        "head.reg_preds head.obj_preds"
    )
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("archive/data.pkl", markers)

    assert detect_version(str(checkpoint)) == YOLOX_CONVERSION


@pytest.mark.parametrize(
    ("head_channels", "expected"),
    [
        (64, (0.33, 0.25, True)),
        (96, (0.33, 0.375, False)),
        (128, (0.33, 0.50, False)),
        (192, (0.67, 0.75, False)),
        (256, (1.00, 1.00, False)),
        (320, (1.33, 1.25, False)),
    ],
)
def test_infers_standard_yolox_architecture(head_channels, expected):
    state_dict = {"head.cls_preds.0.weight": torch.empty(80, head_channels, 1, 1)}

    assert _infer_standard_architecture(state_dict) == expected


def test_yolox_exporter_uses_unscaled_bgr_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_kwargs: dict = {}

    def record_make_nn_archive(_exporter, **kwargs) -> None:
        archive_kwargs.update(kwargs)

    monkeypatch.setattr(
        YoloXExporter,
        "make_nn_archive",
        record_make_nn_archive,
    )

    exporter = object.__new__(YoloXExporter)
    exporter.names = ["person"]
    exporter.nc = 1

    exporter.export_nn_archive()

    assert archive_kwargs["encoding"] == Encoding.BGR
    assert archive_kwargs["mean"] == [0, 0, 0]
    assert archive_kwargs["scale"] == [1, 1, 1]
