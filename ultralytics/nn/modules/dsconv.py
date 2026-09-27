# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# Dynamic Snake Convolution (DSConv) adapted from DSCNet:
#   Qi et al., "Dynamic Snake Convolution based on Topological Geometric
#   Constraints for Tubular Structure Segmentation", ICCV 2023.
#   Reference implementation: https://github.com/YaoleiQi/DSCNet (MIT License)
#
# Fabric-defect motivation (FG-YOLO Stage 2): Broken_yarn defects are extremely
# elongated thin-line structures (measured aspect ratios up to ~85:1) that stock
# square kernels confuse with the periodic weave -> dominant false-positive
# source. DSConv deforms the kernel along such structures. DCFE-YOLO
# (Zhou et al., Sci. Rep. 2025) showed DSConv cuts texture-induced FPs on fabric.
#
# Integration follows the FG-YOLO additive-residual rule: SnakeBlock is wrapped
# as y = x + gamma * f(x) with gamma zero-initialized, so the model starts
# EXACTLY as its pretrained parent and learns snake features only if they help.

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ("DSConv", "SnakeBlock")


class DSConv(nn.Module):
    """
    Dynamic Snake Convolution (2D).

    Learns per-position offsets, accumulates them iteratively outward from the
    kernel center (snake continuity constraint), deforms the feature map by
    bilinear sampling along a 1xK (morph=1) or Kx1 (morph=0) chain, then applies
    the strip convolution.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        k (int): Kernel chain length (number of snake points).
        extend_scope (float): Offset range multiplier (offsets are tanh-bounded to [-1, 1]).
        morph (int): 0 = kernel chain along x-axis, 1 = along y-axis.
    """

    def __init__(self, c1, c2, k=9, extend_scope=1.0, morph=0):
        super().__init__()
        self.k = k
        self.extend_scope = extend_scope
        self.morph = morph
        self.offset_conv = nn.Conv2d(c1, 2 * k, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * k)
        if morph == 0:
            self.dsc_conv = nn.Conv2d(c1, c2, (k, 1), stride=(k, 1))
        else:
            self.dsc_conv = nn.Conv2d(c1, c2, (1, k), stride=(1, k))
        g = max(1, c2 // 4)
        while c2 % g:
            g -= 1
        self.gn = nn.GroupNorm(g, c2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, f):
        offset = torch.tanh(self.bn(self.offset_conv(f)))  # [B, 2K, W, H], in [-1, 1]
        deformed = _deform(f, offset, self.k, self.extend_scope, self.morph)
        return self.act(self.gn(self.dsc_conv(deformed)))


def _coordinate_map(offset, k, extend_scope, morph):
    """Build the deformed sampling coordinates. offset: [B, 2K, W, H]."""
    device = offset.device
    b, _, w, h = offset.shape
    y_offset, x_offset = torch.split(offset, k, dim=1)

    # base coordinate grids, [1, K, W, H]
    y_center = torch.arange(w, device=device).repeat(h).reshape(h, w).permute(1, 0)
    y_center = y_center.reshape(1, w, h).repeat(k, 1, 1).float().unsqueeze(0)
    x_center = torch.arange(h, device=device).repeat(w).reshape(w, h).permute(0, 1)
    x_center = x_center.reshape(1, w, h).repeat(k, 1, 1).float().unsqueeze(0)

    center = k // 2
    if morph == 0:
        # chain runs along x: y is free to snake, x steps -center..center
        spread = torch.linspace(-center, center, k, device=device)
        x_grid = spread.reshape(k, 1, 1).repeat(1, w, h).unsqueeze(0).float()
        y_grid = torch.zeros(1, k, w, h, device=device)
        # iterative offset accumulation outward from the chain center
        yo = y_offset.permute(1, 0, 2, 3)  # [K, B, W, H]
        pts = [None] * k
        pts[center] = torch.zeros_like(yo[0])
        for i in range(1, center + 1):
            pts[center + i] = pts[center + i - 1] + yo[center + i]
            pts[center - i] = pts[center - i + 1] + yo[center - i]
        y_off = torch.stack(pts, 0).permute(1, 0, 2, 3)  # [B, K, W, H]
        y_new = y_center + y_grid + y_off * extend_scope  # [B, K, W, H] via broadcast
        x_new = (x_center + x_grid).repeat(b, 1, 1, 1)
        # -> [B, K*W, H]
        y_new = y_new.reshape(b, k, 1, w, h).permute(0, 3, 1, 4, 2).reshape(b, k * w, h)
        x_new = x_new.reshape(b, k, 1, w, h).permute(0, 3, 1, 4, 2).reshape(b, k * w, h)
    else:
        # chain runs along y: x is free to snake, y steps -center..center
        spread = torch.linspace(-center, center, k, device=device)
        y_grid = spread.reshape(k, 1, 1).repeat(1, w, h).unsqueeze(0).float()
        x_grid = torch.zeros(1, k, w, h, device=device)
        xo = x_offset.permute(1, 0, 2, 3)
        pts = [None] * k
        pts[center] = torch.zeros_like(xo[0])
        for i in range(1, center + 1):
            pts[center + i] = pts[center + i - 1] + xo[center + i]
            pts[center - i] = pts[center - i + 1] + xo[center - i]
        x_off = torch.stack(pts, 0).permute(1, 0, 2, 3)
        y_new = (y_center + y_grid).repeat(b, 1, 1, 1)
        x_new = x_center + x_grid + x_off * extend_scope  # [B, K, W, H] via broadcast
        # -> [B, W, K*H]
        y_new = y_new.reshape(b, 1, k, w, h).permute(0, 3, 1, 4, 2).reshape(b, w, k * h)
        x_new = x_new.reshape(b, 1, k, w, h).permute(0, 3, 1, 4, 2).reshape(b, w, k * h)
    return y_new, x_new


def _bilinear_sample(feat, y, x, w, h, morph):
    """Sample feat [B, C, W, H] at float coords y/x via F.grid_sample.

    CUDA-native replacement for the original manual gather (flat[idx] weights):
    same bilinear math (padding_mode="border" reproduces the old clamp-to-edge),
    but no index arithmetic (no possible out-of-bounds assert), contiguous output
    (avoids cuDNN "no engine" failures on the strip conv), and no giant fp32
    temporaries (fixes the ~9 GiB peak at batch 64 that OOM'd the T4).

    y indexes the dim of size w, x indexes the dim of size h. morph only selects
    the point layout: [B, K*W, H] (morph=0) or [B, W, K*H] (morph=1) — the grid
    is just the stacked coords along a new last dim either way.
    """
    # align_corners=False maps pixel index p -> 2(p+0.5)/size - 1
    gx = (x + 0.5) * (2.0 / h) - 1.0
    gy = (y + 0.5) * (2.0 / w) - 1.0
    grid = torch.stack((gx, gy), dim=-1).to(feat.dtype)  # [B, P, Q, 2]
    return F.grid_sample(feat, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _deform(f, offset, k, extend_scope, morph):
    y, x = _coordinate_map(offset, k, extend_scope, morph)
    return _bilinear_sample(f, y, x, f.shape[2], f.shape[3], morph)


class SnakeBlock(nn.Module):
    """
    Zero-init residual snake feature enhancement (FG-YOLO Stage 2).

    Multi-view snake fusion in the spirit of DCFE-YOLO (x-snake, y-snake,
    standard 3x3 branch concatenated and fused by 1x1), but added residually
    with a learnable gate gamma initialized to 0 — so a pretrained model is
    preserved exactly at init and the snake path is learned only where it helps.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels (must equal c1 for the residual add).
        k (int): Snake kernel chain length.
    """

    def __init__(self, c1, c2, k=9):
        super().__init__()
        self.snake_x = DSConv(c1, c2, k, morph=0)
        self.snake_y = DSConv(c1, c2, k, morph=1)
        self.std = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.fuse = nn.Conv2d(3 * c2, c2, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if self.training and x.requires_grad:
            # DSConv sampling allocates large intermediates; recompute them in
            # backward instead of storing (keeps batch=128 feasible on 16GB GPUs)
            y = self.fuse(
                torch.cat(
                    [
                        checkpoint(self.snake_x, x, use_reentrant=False),
                        checkpoint(self.snake_y, x, use_reentrant=False),
                        checkpoint(self.std, x, use_reentrant=False),
                    ],
                    dim=1,
                )
            )
        else:
            y = self.fuse(torch.cat([self.snake_x(x), self.snake_y(x), self.std(x)], dim=1))
        return x + self.gamma * y
