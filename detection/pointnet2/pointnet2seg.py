import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def square_distance(src, dst):
    batch_size, num_src, _ = src.shape
    _, num_dst, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, dim=-1).view(batch_size, num_src, 1)
    dist += torch.sum(dst**2, dim=-1).view(batch_size, 1, num_dst)
    return dist


def index_points(points, idx):
    device = points.device
    batch_size = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(batch_size, dtype=torch.long, device=device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=device)
    distance = torch.ones(batch_size, num_points, device=device) * 1e10
    farthest = torch.randint(0, num_points, (batch_size,), dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    _, num_centroids, _ = new_xyz.shape
    group_idx = torch.arange(num_points, dtype=torch.long, device=device).view(1, 1, num_points)
    group_idx = group_idx.repeat(batch_size, num_centroids, 1)

    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = num_points
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    first_group = group_idx[:, :, 0].view(batch_size, num_centroids, 1).repeat(1, 1, nsample)
    mask = group_idx == num_points
    group_idx[mask] = first_group[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(xyz.shape[0], npoint, 1, xyz.shape[-1])

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    new_xyz = torch.zeros(batch_size, 1, 3, device=device)
    grouped_xyz = xyz.view(batch_size, 1, num_points, 3)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(batch_size, 1, num_points, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        layers = []
        last_channel = in_channel
        for out_channel in mlp:
            layers.append(nn.Conv2d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU(inplace=True))
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )

        new_points = new_points.permute(0, 3, 2, 1)
        new_points = self.mlp(new_points)
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        layers = []
        last_channel = in_channel
        for out_channel in mlp:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU(inplace=True))
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        batch_size, num_points, _ = xyz1.shape
        _, sampled_points, _ = xyz2.shape

        if sampled_points == 1:
            interpolated_points = points2.repeat(1, num_points, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists = dists[:, :, :3]
            idx = idx[:, :, :3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(batch_size, num_points, 3, 1),
                dim=2,
            )

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        return self.mlp(new_points)


SEG_INPUT_CHANNELS = 9


class PointNet2SemSegSSG(nn.Module):
    def __init__(self, num_classes=13, dropout=0.5):
        super().__init__()
        extra_channels = SEG_INPUT_CHANNELS - 3
        self.sa1 = PointNetSetAbstraction(
            npoint=1024,
            radius=0.1,
            nsample=32,
            in_channel=3 + extra_channels,
            mlp=[32, 32, 64],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=256,
            radius=0.2,
            nsample=32,
            in_channel=64 + 3,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=64,
            radius=0.4,
            nsample=32,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=16,
            radius=0.8,
            nsample=32,
            in_channel=256 + 3,
            mlp=[256, 256, 512],
            group_all=False,
        )
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128 + extra_channels, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != SEG_INPUT_CHANNELS:
            raise ValueError(
                f"Expected input shape [B, {SEG_INPUT_CHANNELS}, N], got {tuple(x.shape)}"
            )
        xyz = x[:, :3, :]
        points = x[:, 3:, :]

        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, points, l1_points)

        features = F.relu(self.bn1(self.conv1(l0_points)), inplace=True)
        features = self.drop1(features)
        logits = self.conv2(features)
        return logits


__all__ = ["PointNet2SemSegSSG"]
