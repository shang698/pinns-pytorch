import math

import torch
from torch import nn


class PINN(nn.Module):
    def __init__(self, widths, lower, upper, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.register_buffer("lower", torch.as_tensor(lower, dtype=dtype, device=device))
        self.register_buffer("upper", torch.as_tensor(upper, dtype=dtype, device=device))
        self.layers = nn.ModuleList(
            [nn.Linear(a, b, dtype=dtype, device=device) for a, b in zip(widths[:-1], widths[1:])]
        )
        for layer in self.layers:
            std = math.sqrt(2.0 / (layer.in_features + layer.out_features))
            nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
            nn.init.zeros_(layer.bias)

    def forward(self, xt):
        h = 2.0 * (xt - self.lower) / (self.upper - self.lower) - 1.0
        for layer in self.layers[:-1]:
            h = torch.tanh(layer(h))
        return self.layers[-1](h)
