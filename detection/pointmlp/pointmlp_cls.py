import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBNReLU1D, LocalGrouper, PosExtraction, PreExtraction, get_activation


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
            anchor_points = max(anchor_points // reducers[i], 1)
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
