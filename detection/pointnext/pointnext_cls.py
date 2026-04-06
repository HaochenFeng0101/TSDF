import torch
import torch.nn as nn


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


class ResidualMLPBlock2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.act(x + identity)
        return x


class PointNeXtSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, out_channel, depth=2, group_all=False):
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

    def forward(self, xyz, points):
        xyz_t = xyz.permute(0, 2, 1)
        points_t = points.permute(0, 2, 1) if points is not None else None

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz_t, points_t)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz_t, points_t)

        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        new_points = self.input_proj(new_points)
        new_points = self.blocks(new_points)
        new_points = torch.max(new_points, dim=2)[0]
        new_xyz = new_xyz.permute(0, 2, 1).contiguous()
        return new_xyz, new_points


class PointNeXtSmallCls(nn.Module):
    """PointNeXt-Small style hierarchical point classifier."""

    def __init__(self, num_classes=40, input_channels=3, dropout=0.4):
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
            radius=0.3,
            nsample=32,
            in_channel=67,
            out_channel=128,
            depth=2,
            group_all=False,
        )
        self.sa3 = PointNeXtSetAbstraction(
            npoint=64,
            radius=0.6,
            nsample=32,
            in_channel=131,
            out_channel=256,
            depth=2,
            group_all=False,
        )
        self.sa4 = PointNeXtSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=259,
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

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] < 3:
            raise ValueError(f"Expected input shape [B, C>=3, N], got {tuple(x.shape)}")

        xyz = x[:, :3, :]
        points = x[:, 3:, :] if x.shape[1] > 3 else None

        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        _, l4_points = self.sa4(l3_xyz, l3_points)

        global_feat = l4_points.squeeze(-1)
        logits = self.head(global_feat)
        return logits


PointNeXtCls = PointNeXtSmallCls
