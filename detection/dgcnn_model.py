import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    return torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()


class DGCNNCls(nn.Module):
    def __init__(self, k=40, num_neighbors=20, emb_dims=1024, dropout=0.5, leaky_relu=True):
        super().__init__()
        self.num_neighbors = num_neighbors

        negative_slope = 0.2 if leaky_relu else 0.0
        activation = (
            nn.LeakyReLU(negative_slope=negative_slope)
            if leaky_relu
            else nn.ReLU(inplace=True)
        )

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                activation,
            )

        self.conv1 = conv_block(6, 64)
        self.conv2 = conv_block(128, 64)
        self.conv3 = conv_block(128, 128)
        self.conv4 = conv_block(256, 256)
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            activation,
        )

        self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, k)
        self.activation = activation

    def forward(self, x):
        x = get_graph_feature(x, k=self.num_neighbors)
        x = self.conv1(x)
        x1 = x.max(dim=-1)[0]

        x = get_graph_feature(x1, k=self.num_neighbors)
        x = self.conv2(x)
        x2 = x.max(dim=-1)[0]

        x = get_graph_feature(x2, k=self.num_neighbors)
        x = self.conv3(x)
        x3 = x.max(dim=-1)[0]

        x = get_graph_feature(x3, k=self.num_neighbors)
        x = self.conv4(x)
        x4 = x.max(dim=-1)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        x_max = F.adaptive_max_pool1d(x, 1).view(x.size(0), -1)
        x_avg = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        x = torch.cat((x_max, x_avg), dim=1)

        x = self.linear1(x)
        x = self.bn6(x)
        x = self.activation(x)
        x = self.dp1(x)

        x = self.linear2(x)
        x = self.bn7(x)
        x = self.activation(x)
        x = self.dp2(x)

        return self.linear3(x)
