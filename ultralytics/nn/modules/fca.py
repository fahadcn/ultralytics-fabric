# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# FcaGate — multi-spectral (DCT) channel gate, FG-YOLO Stage 5.
#
# Motivation (fabric defect FP suppression): standard SE/CBAM attention pools with
# global AVERAGE pooling, which — as FcaNet (Qin et al., CVPR 2021) showed — uses only
# the DC (lowest) frequency component of the feature map. On woven fabric the
# background weave is periodic, so its energy concentrates in specific DCT
# components; a GAP-based gate collapses everything into one weave-dominated mean and
# cannot separate weave channels from defect channels. FcaGate instead pools each
# channel onto K 2D-DCT basis functions and gates channels from that spectrum.
#
# Placement (S5): on the S1b residual skip laterals (P4/P5), before BiFPNAdd. Those
# laterals are random-init in S1b anyway, so no transfer-safety constraint applies.
# Frequency ops at FEATURE level — per the S2a lesson (pixel-level fails).

import torch
from torch import nn

__all__ = ("FcaGate",)


def _dct_basis(h, w, k, device, dtype):
    """K lowest 2D-DCT-II basis functions in (u+v) zigzag order, orthonormalized.

    Returns [K, H, W]; basis[0] is the (scaled) DC component == GAP.
    """
    pairs = sorted(((u, v) for u in range(h) for v in range(w)), key=lambda p: (p[0] + p[1], p[0]))[:k]
    i = torch.arange(h, device=device, dtype=dtype) + 0.5
    j = torch.arange(w, device=device, dtype=dtype) + 0.5
    basis = []
    for u, v in pairs:
        bu = torch.cos(torch.pi * u * i / h) * (1.0 if u == 0 else 2.0) ** 0.5 / h**0.5
        bv = torch.cos(torch.pi * v * j / w) * (1.0 if v == 0 else 2.0) ** 0.5 / w**0.5
        basis.append(torch.outer(bu, bv))
    return torch.stack(basis)


class FcaGate(nn.Module):
    """Multi-spectral frequency channel gate: y = x * sigmoid(FC(ReLU(FC(DCT_pool(x))))).

    Args:
        c (int): Channels (set from the input by tasks.py; c2 == c1).
        k (int): Number of DCT frequency components per channel (FcaNet uses 16).
        r (int): FC reduction ratio.
    """

    def __init__(self, c, k=16, r=4):
        super().__init__()
        self.c, self.k = c, k
        hidden = max(c // r, 8)
        self.fc = nn.Sequential(
            nn.Linear(c * k, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, c),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        w0 = self.fc[0].weight
        # Basis rebuilt fresh on x's device every call (microseconds for 16 small
        # cosine grids). A cached tensor survives neither .to()/deepcopy/EMA —
        # a stale CPU basis caused "mat2 is on cpu" at final_eval (2026-09-30).
        basis = _dct_basis(h, w, self.k, x.device, w0.dtype)
        s = torch.einsum("bchw,khw->bck", x.to(w0.dtype), basis)
        a = self.fc(s.reshape(b, -1)).to(x.dtype).view(b, c, 1, 1)
        return x * a
