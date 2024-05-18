import torch
import torch.nn as nn


class Multiplier(nn.Module):
    def forward(self, prototypes, coefficients):
        coefficients = coefficients.view(coefficients.shape[0], -1, 1, 1)
        x = coefficients * prototypes
        res = torch.sigmoid(x.sum(dim = 1, keepdim = True))
        return res
