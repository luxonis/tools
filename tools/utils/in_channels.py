import torch


def get_first_conv2d_in_channels(model):
    """Get the number of input channels of the first Conv2d layer in the model."""
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            return layer.in_channels
    return None
