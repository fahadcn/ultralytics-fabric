# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
from torch import nn

__all__ = ("BiFPNAdd",)


class BiFPNAdd(nn.Module):
    """
    BiFPN-style learnable weighted feature fusion (fast normalized fusion, Tan et al., CVPR 2020).

    Fuses N input feature maps of identical shape/channels as sum(w_i * x_i) / (sum(w) + eps),
    where w_i = ReLU(theta_i) are learnable scalar weights. Unlike Concat, it adds no channels
    and lets the network learn the relative importance of each input scale.

    Args:
        n_inputs (int): Number of input feature maps to fuse (2 or 3 in the FG-YOLO neck).
        eps (float): Small constant for numerical stability of the normalization.
    """

    def __init__(self, n_inputs=2, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(n_inputs))

    def forward(self, x):
        w = torch.relu(self.w)
        w = w / (w.sum() + self.eps)
        return sum(wi * xi for wi, xi in zip(w, x))
