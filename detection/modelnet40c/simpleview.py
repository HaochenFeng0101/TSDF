import math

import torch
import torch.nn as nn


RESOLUTION = 128
TRANS = -1.4


def euler2mat(angle):
    if angle.dim() == 1:
        x, y, z = angle[0], angle[1], angle[2]
        dim = 0
        view = [3, 3]
    elif angle.dim() == 2:
        batch_size, _ = angle.size()
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]
        dim = 1
        view = [batch_size, 3, 3]
    else:
        raise ValueError(f"Unsupported angle shape: {tuple(angle.shape)}")

    cosz = torch.cos(z)
    sinz = torch.sin(z)
    zero = z.detach() * 0
    one = zero.detach() + 1
    zmat = torch.stack(
        [cosz, -sinz, zero, sinz, cosz, zero, zero, zero, one], dim=dim
    ).reshape(view)

    cosy = torch.cos(y)
    siny = torch.sin(y)
    ymat = torch.stack(
        [cosy, zero, siny, zero, one, zero, -siny, zero, cosy], dim=dim
    ).reshape(view)

    cosx = torch.cos(x)
    sinx = torch.sin(x)
    xmat = torch.stack(
        [one, zero, zero, zero, cosx, -sinx, zero, sinx, cosx], dim=dim
    ).reshape(view)
    return xmat @ ymat @ zmat


def distribute(depth, coord_x, coord_y, size_x, size_y, image_height, image_width):
    batch_size, _ = depth.size()
    epsilon = torch.tensor([1e-12], requires_grad=False, device=depth.device)
    offset_x = torch.linspace(
        -size_x / 2, (size_x / 2) - 1, size_x, requires_grad=False, device=depth.device
    )
    offset_y = torch.linspace(
        -size_y / 2, (size_y / 2) - 1, size_y, requires_grad=False, device=depth.device
    )

    ext_x = coord_x.unsqueeze(2).repeat(1, 1, size_x) + offset_x
    ext_y = coord_y.unsqueeze(2).repeat(1, 1, size_y) + offset_y
    ext_x = ext_x.unsqueeze(3).repeat(1, 1, 1, size_y)
    ext_y = ext_y.unsqueeze(2).repeat(1, 1, size_x, 1)
    ext_x.ceil_()
    ext_y.ceil_()

    value = depth.unsqueeze(2).unsqueeze(3).repeat(1, 1, size_x, size_y)
    valid = (
        (ext_x >= 0)
        * (ext_x <= image_height - 1)
        * (ext_y >= 0)
        * (ext_y <= image_width - 1)
        * (value >= 0)
    )

    ext_x = ext_x % image_height
    ext_y = ext_y % image_width

    weight = valid.float() * (1 / (value + epsilon))
    weighted_value = value * weight

    weight = weight.view(batch_size, -1)
    weighted_value = weighted_value.view(batch_size, -1)
    coordinates = (ext_x.view(batch_size, -1) * image_width) + ext_y.view(batch_size, -1)

    weight_scattered = torch.zeros(
        batch_size, image_width * image_height, device=depth.device
    ).scatter_add(1, coordinates.long(), weight)
    zero_mask = weight_scattered == 0.0
    weight_scattered += zero_mask.float()

    value_scattered = torch.zeros(
        batch_size, image_width * image_height, device=depth.device
    ).scatter_add(1, coordinates.long(), weighted_value)

    return value_scattered, weight_scattered


def points2depth(points, image_height, image_width, size_x=4, size_y=4):
    epsilon = torch.tensor([1e-12], requires_grad=False, device=points.device)
    coord_x = (points[:, :, 0] / (points[:, :, 2] + epsilon)) * (image_width / image_height)
    coord_y = points[:, :, 1] / (points[:, :, 2] + epsilon)
    batch_size, _, _ = points.size()
    depth = points[:, :, 2]
    grid_x = ((coord_x + 1) * image_height) / 2
    grid_y = ((coord_y + 1) * image_width) / 2
    value_scattered, weight_scattered = distribute(
        depth=depth,
        coord_x=grid_x,
        coord_y=grid_y,
        size_x=size_x,
        size_y=size_y,
        image_height=image_height,
        image_width=image_width,
    )
    return (value_scattered / weight_scattered).view(batch_size, image_height, image_width)


class Squeeze(nn.Module):
    def forward(self, x):
        return x.squeeze(-1).squeeze(-1)


class BatchNormPoint(nn.Module):
    def __init__(self, feat_size):
        super().__init__()
        self.feat_size = feat_size
        self.bn = nn.BatchNorm1d(feat_size)

    def forward(self, x):
        batch_size, num_views, feat_size = x.shape
        if feat_size != self.feat_size:
            raise ValueError(f"Expected feat_size={self.feat_size}, got {feat_size}")
        x = x.reshape(batch_size * num_views, feat_size)
        x = self.bn(x)
        return x.view(batch_size, num_views, feat_size)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, feature_size=16):
        super().__init__()
        self.inplanes = feature_size
        self.conv1 = nn.Conv2d(
            3, feature_size, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, feature_size, layers[0])
        self.layer2 = self._make_layer(block, feature_size * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, feature_size * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, feature_size * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(feature_size * 8 * block.expansion, 1000)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class PCViews:
    def __init__(self):
        views = torch.tensor(
            [
                [[0 * math.pi / 2, 0, math.pi / 2], [0, 0, TRANS]],
                [[1 * math.pi / 2, 0, math.pi / 2], [0, 0, TRANS]],
                [[2 * math.pi / 2, 0, math.pi / 2], [0, 0, TRANS]],
                [[3 * math.pi / 2, 0, math.pi / 2], [0, 0, TRANS]],
                [[0, -math.pi / 2, math.pi / 2], [0, 0, TRANS]],
                [[0, math.pi / 2, math.pi / 2], [0, 0, TRANS]],
            ],
            dtype=torch.float32,
        )
        self.num_views = 6
        self.angle = views[:, 0, :]
        self.translation = views[:, 1, :].unsqueeze(1)

    def get_img(self, points):
        batch_size, _, _ = points.shape
        device = points.device
        rot_mat = euler2mat(self.angle.to(device)).transpose(1, 2)
        translation = self.translation.to(device)
        num_views = translation.shape[0]
        transformed = self.point_transform(
            points=torch.repeat_interleave(points, num_views, dim=0),
            rot_mat=rot_mat.repeat(batch_size, 1, 1),
            translation=translation.repeat(batch_size, 1, 1),
        )
        return points2depth(
            points=transformed,
            image_height=RESOLUTION,
            image_width=RESOLUTION,
            size_x=1,
            size_y=1,
        )

    @staticmethod
    def point_transform(points, rot_mat, translation):
        return torch.matmul(points, rot_mat) - translation


class MVFC(nn.Module):
    def __init__(self, num_views, in_features, out_features, dropout_p):
        super().__init__()
        self.num_views = num_views
        self.in_features = in_features
        self.model = nn.Sequential(
            BatchNormPoint(in_features),
            nn.Dropout(dropout_p),
            nn.Flatten(),
            nn.Linear(in_features * num_views, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(in_features, out_features),
        )

    def forward(self, feat):
        feat = feat.view(-1, self.num_views, self.in_features)
        return self.model(feat)


class SimpleViewCls(nn.Module):
    def __init__(self, num_classes=40, feat_size=16, dropout=0.5):
        super().__init__()
        self.dropout_p = dropout
        self.pc_views = PCViews()
        self.num_views = self.pc_views.num_views

        backbone = ResNet(BasicBlock, [2, 2, 2, 2], feature_size=feat_size)
        all_layers = list(backbone.children())
        in_features = all_layers[-1].in_features
        main_layers = all_layers[4:-1]
        self.img_model = nn.Sequential(
            nn.Conv2d(1, feat_size, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feat_size),
            nn.ReLU(inplace=True),
            *main_layers,
            Squeeze(),
        )
        self.final_fc = MVFC(
            num_views=self.num_views,
            in_features=in_features,
            out_features=num_classes,
            dropout_p=self.dropout_p,
        )

    def get_img(self, pc):
        return self.pc_views.get_img(pc).unsqueeze(1)

    def forward(self, pc):
        if pc.dim() != 3:
            raise ValueError(
                f"Expected point cloud shape [B, 3, N] or [B, N, 3], got {tuple(pc.shape)}"
            )
        if pc.shape[1] == 3:
            pc = pc.transpose(1, 2).contiguous()
        img = self.get_img(pc)
        feat = self.img_model(img)
        return self.final_fc(feat)
