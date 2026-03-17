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


class PointMLPModel(nn.Module):
    def __init__(
        self,
        points=1024,
        class_num=40,
        embed_dim=64,
        groups=1,
        res_expansion=1.0,
        activation="relu",
        bias=False,
        use_xyz=False,
        normalize="anchor",
        dim_expansion=(2, 2, 2, 2),
        pre_blocks=(2, 2, 2, 2),
        pos_blocks=(2, 2, 2, 2),
        k_neighbors=(24, 24, 24, 24),
        reducers=(2, 2, 2, 2),
    ):
        super().__init__()
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)

        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion)

        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()

        last_channel = embed_dim
        anchor_points = self.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            anchor_points = anchor_points // reducers[i]
            self.local_grouper_list.append(
                LocalGrouper(last_channel, anchor_points, k_neighbors[i], use_xyz, normalize)
            )
            self.pre_blocks_list.append(
                PreExtraction(
                    last_channel,
                    out_channel,
                    pre_blocks[i],
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                    use_xyz=use_xyz,
                )
            )
            self.pos_blocks_list.append(
                PosExtraction(
                    out_channel,
                    pos_blocks[i],
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                )
            )
            last_channel = out_channel

        act = get_activation(activation)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            act,
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            act,
            nn.Dropout(0.5),
            nn.Linear(256, self.class_num),
        )

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        x = self.embedding(x)
        for i in range(self.stages):
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))
            x = self.pre_blocks_list[i](x)
            x = self.pos_blocks_list[i](x)
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        return self.classifier(x)


def pointMLP(num_classes=40, points=1024, **kwargs):
    return PointMLPModel(
        points=points,
        class_num=num_classes,
        embed_dim=64,
        groups=1,
        res_expansion=1.0,
        activation="relu",
        bias=False,
        use_xyz=False,
        normalize="anchor",
        dim_expansion=(2, 2, 2, 2),
        pre_blocks=(2, 2, 2, 2),
        pos_blocks=(2, 2, 2, 2),
        k_neighbors=(24, 24, 24, 24),
        reducers=(2, 2, 2, 2),
        **kwargs,
    )


def pointMLPElite(num_classes=40, points=1024, **kwargs):
    return PointMLPModel(
        points=points,
        class_num=num_classes,
        embed_dim=32,
        groups=1,
        res_expansion=0.25,
        activation="relu",
        bias=False,
        use_xyz=False,
        normalize="anchor",
        dim_expansion=(2, 2, 2, 1),
        pre_blocks=(1, 1, 2, 1),
        pos_blocks=(1, 1, 2, 1),
        k_neighbors=(24, 24, 24, 24),
        reducers=(2, 2, 2, 2),
        **kwargs,
    )


class PointMLPCls(nn.Module):
    def __init__(self, k=40, num_points=1024, model_type="pointmlp", **kwargs):
        super().__init__()
        factory = pointMLPElite if model_type.lower() == "pointmlpelite" else pointMLP
        self.model = factory(num_classes=k, points=num_points, **kwargs)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    data = torch.rand(2, 3, 1024)
    model = pointMLP(num_classes=15)
    out = model(data)
    print(out.shape)
