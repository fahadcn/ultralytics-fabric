# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
from torch import nn

from ultralytics.nn.modules.conv import Conv

__all__ = ("WTDown",)


class WTDown(nn.Module):
    """
    Haar wavelet downsampling (stride 2) — FG-YOLO Stage 2.

    Instead of a learned strided convolution (which low-passes and destroys the
    high-frequency cues that thin fabric defects depend on), the input is split
    into the four fixed Haar sub-bands (LL, LH, HL, HH) at half resolution,
    stacked channel-wise, and mixed by a 1x1 Conv+BN+SiLU. No information is
    discarded before the first learned filter: the network sees low-pass
    structure AND horizontal/vertical/diagonal high-frequency detail explicitly.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
    """

    def __init__(self, c1, c2):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, 1, 1)

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 0::2, 1::2]
        x2 = x[:, :, 1::2, 0::2]
        x3 = x[:, :, 1::2, 1::2]
        ll = (x0 + x1 + x2 + x3) / 2
        lh = (x0 - x1 + x2 - x3) / 2
        hl = (x0 + x1 - x2 - x3) / 2
        hh = (x0 - x1 - x2 + x3) / 2
        return self.conv(torch.cat([ll, lh, hl, hh], dim=1))
