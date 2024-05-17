from __future__ import annotations

from torch import nn


class YoloV6BackBone(nn.Module):
    """Backbone of YoloV6 model, it takes the model's original backbone and wraps it in
    this universal class.

    This was created for backwards compatibility with R2 models.
    """

    def __init__(
        self, old_layer, uses_fuse_P2: bool = True, uses_6_erblock: bool = False
    ):
        super().__init__()

        self.uses_fuse_P2 = uses_fuse_P2
        self.uses_6_erblock = uses_6_erblock

        self.fuse_P2 = old_layer.fuse_P2 if hasattr(old_layer, "fuse_P2") else False

        self.stem = old_layer.stem
        self.ERBlock_2 = old_layer.ERBlock_2
        self.ERBlock_3 = old_layer.ERBlock_3
        self.ERBlock_4 = old_layer.ERBlock_4
        self.ERBlock_5 = old_layer.ERBlock_5
        if uses_6_erblock:
            self.ERBlock_6 = old_layer.ERBlock_6

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.uses_fuse_P2 and self.fuse_P2:
            outputs.append(x)
        elif not self.uses_fuse_P2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)
        if self.uses_6_erblock:
            x = self.ERBlock_6(x)
            outputs.append(x)

        return tuple(outputs)
