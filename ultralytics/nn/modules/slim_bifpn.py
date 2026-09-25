from torch import nn


class DepthwiseSeparableConv(nn.Module):
    """Lightweight depthwise separable convolution.

    Reduces parameters by ~8x vs standard conv.

    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        k (int): Kernel size
        s (int): Stride
    """

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()

        self.dw = nn.Conv2d(c1, c1, k, s, k // 2, groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.act = nn.SiLU(inplace=True)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x

    def forward_fuse(self, x):
        """Fused inference (no BN)."""
        x = self.act(self.dw(x))
        x = self.act(self.pw(x))
        return x
