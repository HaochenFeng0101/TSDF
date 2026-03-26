import argparse
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import open3d as o3d
except Exception:
    o3d = None


REPO_ROOT = Path(__file__).resolve().parents[2]
TSDF_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSDF.detection.pointnet_model import PointNetCls
from TSDF.detection.train_pointnet_cls import load_point_cloud_file, normalize_points
from TSDF.dataset.modelnet40_data import load_off_points


def load_labels(labels_path):
    with open(labels_path, "r", encoding="utf-8") as handle:
        labels = [line.strip() for line in handle if line.strip()]
    if not labels:
        raise ValueError(f"No labels found in {labels_path}")
    return labels


def load_checkpoint(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        raise RuntimeError(f"Missing checkpoint keys: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    return checkpoint


def sample_points(points, num_points, seed):
    rng = np.random.default_rng(seed)
    if len(points) >= num_points:
        indices = rng.choice(len(points), num_points, replace=False)
    else:
        indices = rng.choice(len(points), num_points, replace=True)
    return points[indices].astype(np.float32)


def prepare_points(points, num_points, seed, use_all_points=False):
    if use_all_points:
        return normalize_points(points.astype(np.float32))
    sampled = sample_points(points, num_points, seed)
    return normalize_points(sampled)


def colorize_points(points):
    mins = points.min(axis=0, keepdims=True)
    maxs = points.max(axis=0, keepdims=True)
    span = np.maximum(maxs - mins, 1e-6)
    colors = (points - mins) / span
    return np.clip(colors, 0.0, 1.0)


def visualize_point_cloud(points, title):
    if o3d is None:
        raise RuntimeError("Open3D is not available, so visualization cannot be shown.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colorize_points(points).astype(np.float64))
    o3d.visualization.draw_geometries([pcd], window_name=title)


def load_input_points(path, num_points, seed, modelnet40_sample_method):
    path = Path(path)
    if path.suffix.lower() == ".off":
        rng = np.random.default_rng(seed)
        return load_off_points(
            path,
            num_points=num_points,
            rng=rng,
            sample_method=modelnet40_sample_method,
        )
    return load_point_cloud_file(path)


def main():
    parser = argparse.ArgumentParser(
        description="Run a trained PointNet checkpoint on one point cloud or ModelNet40 .off file."
    )
    parser.add_argument("--input", required=True, help="Input point file: .off/.pcd/.ply/.npy/.npz/.txt/.xyz/.pts")
    parser.add_argument(
        "--checkpoint",
        default=str(TSDF_ROOT / "model" / "pointnet" / "pointnet_best.pth"),
        help="PointNet checkpoint path.",
    )
    parser.add_argument(
        "--labels",
        default=str(TSDF_ROOT / "model" / "pointnet" / "labels.txt"),
        help="Label file, one class per line.",
    )
    parser.add_argument("--num-points", type=int, default=None, help="Override num_points from checkpoint.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-all-points", action="store_true")
    parser.add_argument(
        "--modelnet40-sample-method",
        choices=["surface", "vertices"],
        default="surface",
        help="How to convert a ModelNet40 .off mesh into point samples.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--visualize", action="store_true", help="Show an Open3D window for the evaluated points.")
    parser.add_argument(
        "--visualize-raw-points",
        action="store_true",
        help="When visualizing, show raw points instead of normalized/evaluated points.",
    )
    args = parser.parse_args()

    labels = load_labels(args.labels)
    model = PointNetCls(k=len(labels)).to(args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, args.device)
    model.eval()

    num_points = args.num_points or checkpoint.get("num_points", 1024)
    raw_points = load_input_points(
        args.input,
        num_points=num_points,
        seed=args.seed,
        modelnet40_sample_method=args.modelnet40_sample_method,
    )
    prepared = prepare_points(
        raw_points,
        num_points=num_points,
        seed=args.seed,
        use_all_points=args.use_all_points,
    )
    tensor = torch.from_numpy(prepared.T).unsqueeze(0).to(device=args.device)
    with torch.no_grad():
        logits, _ = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())

    topk = min(5, len(labels))
    top_probs, top_indices = torch.topk(probs, k=topk)
    print(f"input: {args.input}")
    print(f"raw_num_points: {len(raw_points)}")
    print(f"predicted: {labels[pred_idx]}")
    print(f"confidence: {float(probs[pred_idx]):.4f}")
    print("top_predictions:")
    for rank in range(topk):
        cls_idx = int(top_indices[rank].item())
        score = float(top_probs[rank].item())
        print(f"  {rank + 1}. {labels[cls_idx]} ({score:.4f})")

    if args.visualize:
        points_to_show = raw_points if args.visualize_raw_points else prepared
        title = f"PointNet | pred={labels[pred_idx]}"
        visualize_point_cloud(points_to_show, title)


if __name__ == "__main__":
    main()
