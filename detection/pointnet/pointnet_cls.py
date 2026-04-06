import torch
import torch.nn as nn
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = torch.max(x, 2, keepdim=False)[0]
        x = F.relu(self.bn4(self.fc1(x)), inplace=True)
        x = F.relu(self.bn5(self.fc2(x)), inplace=True)
        x = self.fc3(x)

        identity = torch.eye(3, dtype=x.dtype, device=x.device).view(1, 9)
        x = x + identity.repeat(batch_size, 1)
        return x.view(-1, 3, 3)


class STNkd(nn.Module):
    def __init__(self, k=64):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = torch.max(x, 2, keepdim=False)[0]
        x = F.relu(self.bn4(self.fc1(x)), inplace=True)
        x = F.relu(self.bn5(self.fc2(x)), inplace=True)
        x = self.fc3(x)

        identity = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, -1)
        x = x + identity.repeat(batch_size, 1)
        return x.view(-1, self.k, self.k)


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=True):
        super().__init__()
        self.global_feat = global_feat
        self.stn = STN3d()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fstn = STNkd(k=64) if feature_transform else None

    def forward(self, x):
        num_points = x.size(2)
        trans = self.stn(x)
        x = torch.bmm(trans, x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)

        trans_feat = None
        if self.fstn is not None:
            trans_feat = self.fstn(x)
            x = torch.bmm(trans_feat, x)

        point_features = x
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)

        if self.global_feat:
            return x, trans, trans_feat

        x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
        return torch.cat([x, point_features], dim=1), trans, trans_feat


class PointNetCls(nn.Module):
    def __init__(self, k=40, feature_transform=True, dropout=0.4):
        super().__init__()
        self.k = k
        self.feat = PointNetEncoder(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=dropout)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x, _, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)), inplace=True)
        x = F.relu(self.bn2(self.fc2(x)), inplace=True)
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits, trans_feat


def feature_transform_regularizer(trans):
    if trans is None:
        return 0.0
    dim = trans.size(1)
    identity = torch.eye(dim, dtype=trans.dtype, device=trans.device).unsqueeze(0)
    diff = torch.bmm(trans, trans.transpose(2, 1)) - identity
    return torch.mean(torch.norm(diff, dim=(1, 2)))
