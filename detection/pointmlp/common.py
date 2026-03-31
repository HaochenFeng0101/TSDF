import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pointnet2_ops import pointnet2_utils  # type: ignore
except Exception:
    pointnet2_utils = None


def get_activation(activation):
    activation = activation.lower()
    if activation == "gelu":
        return nn.GELU()
    if activation == "rrelu":
        return nn.RReLU(inplace=True)
    if activation == "selu":
        return nn.SELU(inplace=True)
    if activation == "silu":
        return nn.SiLU(inplace=True)
    if activation == "hardswish":
        return nn.Hardswish(inplace=True)
    if activation == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    return nn.ReLU(inplace=True)


def square_distance(src, dst):
    b, n, _ = src.shape
    _, m, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(b, n, 1)
    dist += torch.sum(dst**2, -1).view(b, 1, m)
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
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def furthest_point_sample(xyz, npoint):
    if pointnet2_utils is not None and xyz.is_cuda:
        return pointnet2_utils.furthest_point_sample(xyz, npoint).long()
    return farthest_point_sample(xyz, npoint).long()


def knn_point(nsample, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="center"):
        super().__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        self.normalize = normalize.lower() if normalize is not None else None
        if self.normalize not in ["center", "anchor", None]:
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        batch_size, _, _ = xyz.shape
        groups = self.groups
        fps_idx = furthest_point_sample(xyz, groups)
        new_xyz = index_points(xyz, fps_idx)
        new_points = index_points(points, fps_idx)

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_points = index_points(points, idx)

        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)

        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            else:
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)
            std = (
                torch.std((grouped_points - mean).reshape(batch_size, -1), dim=-1, keepdim=True)
                .unsqueeze(dim=-1)
                .unsqueeze(dim=-1)
            )
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta

        new_points = torch.cat(
            [
                grouped_points,
                new_points.view(batch_size, groups, 1, -1).repeat(1, 1, self.kneighbors, 1),
            ],
            dim=-1,
        )
        return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation="relu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            get_activation(activation),
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(
        self,
        channel,
        kernel_size=1,
        groups=1,
        res_expansion=1.0,
        bias=True,
        activation="relu",
    ):
        super().__init__()
        act = get_activation(activation)
        inner = int(channel * res_expansion)
        self.net1 = nn.Sequential(
            nn.Conv1d(channel, inner, kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(inner),
            act,
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(inner, channel, kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                act,
                nn.Conv1d(channel, channel, kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(inner, channel, kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        self.act = act

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        blocks=1,
        groups=1,
        res_expansion=1,
        bias=True,
        activation="relu",
        use_xyz=True,
    ):
        super().__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        ops = []
        for _ in range(blocks):
            ops.append(
                ConvBNReLURes1D(
                    out_channels,
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                )
            )
        self.operation = nn.Sequential(*ops)

    def forward(self, x):
        b, n, s, d = x.size()
        x = x.permute(0, 1, 3, 2).reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        return x.reshape(b, n, -1).permute(0, 2, 1)


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation="relu"):
        super().__init__()
        ops = []
        for _ in range(blocks):
            ops.append(
                ConvBNReLURes1D(
                    channels,
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                )
            )
        self.operation = nn.Sequential(*ops)

    def forward(self, x):
        return self.operation(x)


def interpolate_points(xyz1, xyz2, points2):
    xyz1 = xyz1.permute(0, 2, 1)
    xyz2 = xyz2.permute(0, 2, 1)
    points2 = points2.permute(0, 2, 1)

    batch_size, num_points, _ = xyz1.shape
    _, sampled_points, _ = xyz2.shape

    if sampled_points == 1:
        interpolated = points2.repeat(1, num_points, 1)
    else:
        dists = square_distance(xyz1, xyz2)
        dists, idx = dists.sort(dim=-1)
        dists = dists[:, :, :3]
        idx = idx[:, :, :3]

        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated = torch.sum(
            index_points(points2, idx) * weight.view(batch_size, num_points, 3, 1),
            dim=2,
        )
    return interpolated.permute(0, 2, 1)


class PointMLPFeaturePropagation(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        blocks=1,
        groups=1,
        res_expansion=1.0,
        bias=True,
        activation="relu",
    ):
        super().__init__()
        self.fuse = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        self.post = PosExtraction(
            out_channels,
            blocks=blocks,
            groups=groups,
            res_expansion=res_expansion,
            bias=bias,
            activation=activation,
        )

    def forward(self, xyz1, xyz2, points1, points2):
        interpolated = interpolate_points(xyz1, xyz2, points2)
        if points1 is not None:
            new_points = torch.cat([points1, interpolated], dim=1)
        else:
            new_points = interpolated
        new_points = self.fuse(new_points)
        return self.post(new_points)
