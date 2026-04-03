from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSDF.detection.pointnet2.pointnet2 import sample_and_group, sample_and_group_all


class ResidualMLPBlock2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.act(x + identity)
        return x


class PointNeXtSetAbstraction(nn.Module):
    def __init__(
        self,
        npoint,
        radius,
        nsample,
        in_channel: int,
        out_channel: int,
        depth: int = 2,
        group_all: bool = False,
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualMLPBlock2D(out_channel) for _ in range(max(depth, 1))])

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None):
        # xyz: [B, 3, N], points: [B, D, N] or None
        xyz_t = xyz.permute(0, 2, 1)
        points_t = points.permute(0, 2, 1) if points is not None else None

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz_t, points_t)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint,
                self.radius,
                self.nsample,
                xyz_t,
                points_t,
            )

        # [B, npoint, nsample, C] -> [B, C, nsample, npoint]
        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        new_points = self.input_proj(new_points)
        new_points = self.blocks(new_points)

        # Local max pool over neighbors
        new_points = torch.max(new_points, dim=2)[0]  # [B, C, npoint]
        new_xyz = new_xyz.permute(0, 2, 1).contiguous()  # [B, 3, npoint]
        return new_xyz, new_points


class PointNeXtSmallCls(nn.Module):
    """PointNeXt-Small style hierarchical point classifier."""

    def __init__(self, num_classes: int = 40, input_channels: int = 3, dropout: float = 0.5):
        super().__init__()
        if input_channels < 3:
            raise ValueError("input_channels must be >= 3 (xyz required)")

        self.sa1 = PointNeXtSetAbstraction(
            npoint=512,
            radius=0.15,
            nsample=32,
            in_channel=input_channels,
            out_channel=64,
            depth=2,
            group_all=False,
        )
        self.sa2 = PointNeXtSetAbstraction(
            npoint=256,
            radius=0.30,
            nsample=32,
            in_channel=64 + 3,
            out_channel=128,
            depth=2,
            group_all=False,
        )
        self.sa3 = PointNeXtSetAbstraction(
            npoint=64,
            radius=0.60,
            nsample=32,
            in_channel=128 + 3,
            out_channel=256,
            depth=2,
            group_all=False,
        )
        self.sa4 = PointNeXtSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,
            out_channel=512,
            depth=2,
            group_all=True,
        )

        self.head = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N]
        if x.ndim != 3 or x.shape[1] < 3:
            raise ValueError(f"Expected input shape [B, C>=3, N], got {tuple(x.shape)}")

        xyz = x[:, :3, :]
        points = x[:, 3:, :] if x.shape[1] > 3 else None

        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        _, l4_points = self.sa4(l3_xyz, l3_points)

        # group_all produces npoint=1
        global_feat = l4_points.squeeze(-1)
        logits = self.head(global_feat)
        return logits


PointNeXtCls = PointNeXtSmallCls

__all__ = ["PointNeXtCls", "PointNeXtSmallCls"]
