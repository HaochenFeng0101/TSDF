import sys
from pathlib import Path

import torch
import torch.nn as nn


TSDF_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = TSDF_ROOT / "third_party" / "ModelNet40-C"

if str(TSDF_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(TSDF_ROOT.parent))

from TSDF.detection.modelnet40c.simpleview import SimpleViewCls
from TSDF.detection.pointmlp.model import PointMLPCls
from TSDF.detection.pointnet_model import PointNetCls


MODEL_SPECS = {
    "curvenet": {"label": "CurveNet", "default_loss": "smooth"},
    "pointnet2": {"label": "PointNet++", "default_loss": "cross_entropy"},
    "pct": {"label": "PCT", "default_loss": "smooth"},
    "gdanet": {"label": "GDANet", "default_loss": "smooth"},
    "dgcnn": {"label": "DGCNN", "default_loss": "smooth"},
    "rscnn": {"label": "RSCNN", "default_loss": "cross_entropy"},
    "simpleview": {"label": "SimpleView", "default_loss": "cross_entropy"},
    "pointnet": {"label": "PointNet", "default_loss": "cross_entropy"},
    "pointmlp": {"label": "PointMLP", "default_loss": "smooth"},
    "pointmlpelite": {"label": "PointMLP-Elite", "default_loss": "smooth"},
}

MODEL_ALIASES = {
    "pointnet++": "pointnet2",
    "pointmlp-elite": "pointmlpelite",
    "pointmlp_elite": "pointmlpelite",
}


def ensure_official_root():
    if not OFFICIAL_ROOT.exists():
        raise FileNotFoundError(
            f"Official model source not found: {OFFICIAL_ROOT}. "
            "Clone the upstream ModelNet40-C repo first."
        )
    if str(OFFICIAL_ROOT) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_ROOT))


def normalize_model_name(name):
    raw = name.strip().lower()
    if raw in MODEL_ALIASES:
        return MODEL_ALIASES[raw]
    normalized = raw.replace("_", "").replace("-", "")
    if normalized == "pointmlpelite":
        return "pointmlpelite"
    if normalized == "pointnet++":
        return "pointnet2"
    return normalized


class CurveNetCls(nn.Module):
    def __init__(self, num_classes=40, curve_k=20, curve_setting="default"):
        super().__init__()
        ensure_official_root()
        from CurveNet.core.models.curvenet_cls import CurveNet

        self.model = CurveNet(num_classes=num_classes, k=curve_k, setting=curve_setting)

    def forward(self, x):
        return self.model(x)


class PointNet2Cls(nn.Module):
    def __init__(self, num_classes=40, version_cls=1.0):
        super().__init__()
        ensure_official_root()
        from pointnet2_pyt.pointnet2.models.pointnet2_msg_cls import Pointnet2MSG

        self.model = Pointnet2MSG(
            num_classes=num_classes,
            input_channels=0,
            use_xyz=True,
            version=version_cls,
        )

    def forward(self, x):
        return self.model(x.transpose(1, 2).contiguous())


class PctCls(nn.Module):
    def __init__(self, num_classes=40, dropout=0.5):
        super().__init__()
        ensure_official_root()
        from PCT_Pytorch.model import Pct

        class Args:
            def __init__(self, dropout_value):
                self.dropout = dropout_value

        self.model = Pct(Args(dropout), output_channels=num_classes)

    def forward(self, x):
        return self.model(x)


class GDANetCls(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        ensure_official_root()
        from GDANet.model.GDANet_cls import GDANET

        self.model = GDANET(number_class=num_classes)

    def forward(self, x):
        return self.model(x)


def _knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def _get_graph_feature_device_aware(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = _knn(x, k=k)
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
    def __init__(self, num_classes=40, k=20, emb_dims=1024, dropout=0.5, leaky_relu=True):
        super().__init__()
        ensure_official_root()
        import dgcnn.pytorch.model as dgcnn_model

        dgcnn_model.get_graph_feature = _get_graph_feature_device_aware

        class Args:
            def __init__(self):
                self.k = k
                self.emb_dims = emb_dims
                self.dropout = dropout
                self.leaky_relu = int(leaky_relu)

        self.model = dgcnn_model.DGCNN(Args(), output_channels=num_classes)

    def forward(self, x):
        return self.model(x)


class RSCNNCls(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        ensure_official_root()
        from rs_cnn.models import RSCNN_SSN_Cls

        self.model = RSCNN_SSN_Cls(
            num_classes=num_classes,
            input_channels=0,
            relation_prior=1,
            use_xyz=True,
        )

    def forward(self, x):
        return self.model(x.transpose(1, 2).contiguous())


class PointNetLocalCls(nn.Module):
    def __init__(self, num_classes=40, dropout=0.3):
        super().__init__()
        self.model = PointNetCls(k=num_classes, dropout=dropout)

    def forward(self, x):
        return self.model(x)


class PointMLPLocalCls(nn.Module):
    def __init__(self, num_classes=40, num_points=1024, model_type="pointmlp"):
        super().__init__()
        self.model = PointMLPCls(k=num_classes, num_points=num_points, model_type=model_type)

    def forward(self, x):
        return self.model(x)


def build_model(model_name, num_classes, num_points=1024, simpleview_feat_size=16):
    model_key = normalize_model_name(model_name)
    if model_key not in MODEL_SPECS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {', '.join(sorted(MODEL_SPECS))}"
        )

    if model_key == "curvenet":
        return CurveNetCls(num_classes=num_classes)
    if model_key == "pointnet2":
        return PointNet2Cls(num_classes=num_classes)
    if model_key == "pct":
        return PctCls(num_classes=num_classes)
    if model_key == "gdanet":
        return GDANetCls(num_classes=num_classes)
    if model_key == "dgcnn":
        return DGCNNCls(num_classes=num_classes)
    if model_key == "rscnn":
        return RSCNNCls(num_classes=num_classes)
    if model_key == "simpleview":
        return SimpleViewCls(num_classes=num_classes, feat_size=simpleview_feat_size)
    if model_key == "pointnet":
        return PointNetLocalCls(num_classes=num_classes)
    if model_key == "pointmlp":
        return PointMLPLocalCls(
            num_classes=num_classes, num_points=num_points, model_type="pointmlp"
        )
    if model_key == "pointmlpelite":
        return PointMLPLocalCls(
            num_classes=num_classes, num_points=num_points, model_type="pointmlpelite"
        )
    raise AssertionError("Unreachable model registry branch")
