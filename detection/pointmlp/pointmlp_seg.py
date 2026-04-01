import torch
import torch.nn as nn

from .common import (
    ConvBNReLU1D,
    LocalGrouper,
    PointMLPFeaturePropagation,
    PosExtraction,
    PreExtraction,
)


SEG_INPUT_CHANNELS = 9


class PointMLPSemSegModel(nn.Module):
    def __init__(
        self,
        points=4096,
        class_num=13,
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
        fp_blocks=(1, 1, 1, 1),
        k_neighbors=(24, 24, 24, 24),
        reducers=(2, 2, 2, 2),
        decoder_channels=(512, 256, 128, 128),
        dropout=0.5,
    ):
        super().__init__()
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points
        self.embedding = ConvBNReLU1D(SEG_INPUT_CHANNELS, embed_dim, bias=bias, activation=activation)

        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion)
        assert len(fp_blocks) == len(pre_blocks) == len(decoder_channels)

        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()

        encoder_channels = [embed_dim]
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
            encoder_channels.append(last_channel)

        self.fp_layers = nn.ModuleList()
        current_channel = encoder_channels[-1]
        for stage_idx in range(self.stages - 1, -1, -1):
            skip_channel = encoder_channels[stage_idx]
            out_channel = decoder_channels[self.stages - 1 - stage_idx]
            self.fp_layers.append(
                PointMLPFeaturePropagation(
                    in_channels=current_channel + skip_channel,
                    out_channels=out_channel,
                    blocks=fp_blocks[self.stages - 1 - stage_idx],
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                )
            )
            current_channel = out_channel

        self.classifier = nn.Sequential(
            ConvBNReLU1D(current_channel, 128, bias=bias, activation=activation),
            nn.Dropout(dropout),
            nn.Conv1d(128, self.class_num, kernel_size=1, bias=True),
        )

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != SEG_INPUT_CHANNELS:
            raise ValueError(
                f"Expected input shape [B, {SEG_INPUT_CHANNELS}, N], got {tuple(x.shape)}"
            )

        xyz = x[:, :3, :]
        points = self.embedding(x)

        xyz_stack = [xyz]
        feature_stack = [points]
        current_xyz = xyz.permute(0, 2, 1)
        current_points = points

        for i in range(self.stages):
            current_xyz, grouped = self.local_grouper_list[i](current_xyz, current_points.permute(0, 2, 1))
            current_points = self.pre_blocks_list[i](grouped)
            current_points = self.pos_blocks_list[i](current_points)
            xyz_stack.append(current_xyz.permute(0, 2, 1))
            feature_stack.append(current_points)

        upsampled = feature_stack[-1]
        for layer_idx, fp_layer in enumerate(self.fp_layers):
            coarse_idx = self.stages - layer_idx
            fine_idx = coarse_idx - 1
            upsampled = fp_layer(
                xyz_stack[fine_idx],
                xyz_stack[coarse_idx],
                feature_stack[fine_idx],
                upsampled,
            )
        return self.classifier(upsampled)


def pointMLPSeg(num_classes=13, points=4096, dropout=0.5, **kwargs):
    return PointMLPSemSegModel(
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
        fp_blocks=(1, 1, 1, 1),
        k_neighbors=(24, 24, 24, 24),
        reducers=(2, 2, 2, 2),
        decoder_channels=(512, 256, 128, 128),
        dropout=dropout,
        **kwargs,
    )


def pointMLPEliteSeg(num_classes=13, points=4096, dropout=0.5, **kwargs):
    return PointMLPSemSegModel(
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
        fp_blocks=(1, 1, 1, 1),
        k_neighbors=(24, 24, 24, 24),
        reducers=(2, 2, 2, 2),
        decoder_channels=(256, 128, 96, 96),
        dropout=dropout,
        **kwargs,
    )


class PointMLPSemSeg(nn.Module):
    def __init__(self, num_classes=13, num_points=4096, model_type="pointmlp", dropout=0.5, **kwargs):
        super().__init__()
        factory = pointMLPEliteSeg if model_type.lower() == "pointmlpelite" else pointMLPSeg
        self.model = factory(num_classes=num_classes, points=num_points, dropout=dropout, **kwargs)

    def forward(self, x):
        return self.model(x)
