from __future__ import annotations

import torch


def get_first_conv2d_in_channels(model):
    """Return the input channel count of the first ``Conv2d`` layer.

    Args:
        model: PyTorch model to inspect.

    Returns:
        The number of input channels for the first convolution layer, or
        ``None`` if the model has no ``Conv2d`` layers.
    """
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            return layer.in_channels
    return None
