import argparse
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import open3d as o3d
except Exception:
    o3d = None

'''
python python detection/visualize_pointmlp_pcd.py \
  model/pointmlp/pointmlp_best.pth \
  3d_construction/outputs/chair.pcd


'''

REPO_ROOT = Path(__file__).resolve().parents[2]
TSDF_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSDF.detection.pointmlp.pointmlp_cls import PointMLPCls


def resolve_default_pointmlp_artifact(filename):
    candidates = [
        TSDF_ROOT / "model" / "pointmlp" / filename,
        TSDF_ROOT / "model" / "pointmlp0405" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    model_root = TSDF_ROOT / "model"
    if model_root.exists():
        discovered = sorted(model_root.glob(f"pointmlp*/{filename}"))
        if discovered:
            return discovered[0]

    return candidates[0]


def set_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_labels(labels_path):
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find label file: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        labels = [line.strip() for line in handle if line.strip()]
    if not labels:
        raise ValueError(f"Label file is empty: {path}")
    return labels


def load_checkpoint(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        raise RuntimeError(f"Checkpoint is missing keys: {missing}")
    if unexpected:
        raise RuntimeError(f"Checkpoint has unexpected keys: {unexpected}")
    return checkpoint


def load_point_cloud_points(point_cloud_path):
    if o3d is None:
        raise RuntimeError("Open3D is not available in this environment.")

    path = Path(point_cloud_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find point cloud file: {path}")

    point_cloud = o3d.io.read_point_cloud(str(path))
    if point_cloud.is_empty():
        raise RuntimeError(f"Point cloud is empty or unreadable: {path}")

    points = np.asarray(point_cloud.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"Expected Nx3 points in {path}, got shape {points.shape}")
    return points


def prepare_points(points, num_points, seed, use_all_points=False):
    points = points.astype(np.float32)
    if use_all_points:
        sampled = points
    else:
        rng = np.random.default_rng(seed)
        if len(points) >= num_points:
            indices = rng.choice(len(points), num_points, replace=False)
        else:
            indices = rng.choice(len(points), num_points, replace=True)
        sampled = points[indices]

    centroid = sampled.mean(axis=0, keepdims=True)
    sampled = sampled - centroid
    scale = np.linalg.norm(sampled, axis=1).max()
    if scale > 0:
        sampled = sampled / scale
    return sampled


def predict_with_votes(model, points, device, num_points, num_votes, use_all_points=False):
    logits_votes = []
    for vote_idx in range(max(num_votes, 1)):
        prepared = prepare_points(points, num_points=num_points, seed=vote_idx, use_all_points=use_all_points)
        tensor = torch.from_numpy(prepared.T).unsqueeze(0).to(device=device)
        with torch.no_grad():
            logits = model(tensor)
        logits_votes.append(logits[0].detach().cpu())

    mean_logits = torch.stack(logits_votes, dim=0).mean(dim=0)
    probs = torch.softmax(mean_logits, dim=0)
    pred = int(torch.argmax(probs).item())
    return pred, probs


def inspect_point_cloud(model, points, labels, num_points, device, num_votes, use_all_points=False, topk=3):
    pred, probs = predict_with_votes(
        model=model,
        points=points,
        device=device,
        num_points=num_points,
        num_votes=num_votes,
        use_all_points=use_all_points,
    )

    topk = min(topk, len(labels))
    top_probs, top_indices = torch.topk(probs, k=topk)
    print(f"raw_num_points: {len(points)}")
    print(f"evaluation_mode: {'all_points' if use_all_points else f'{num_votes}_vote_sampling'}")
    print(f"predicted: {labels[pred]}")
    print(f"confidence: {float(top_probs[0]):.4f}")
    print("top_predictions:")
    for rank in range(topk):
        cls_idx = int(top_indices[rank].item())
        score = float(top_probs[rank].item())
        print(f"  {rank + 1}. {labels[cls_idx]} ({score:.4f})")

    return {
        "raw_points": points,
        "prepared_points": prepare_points(
            points,
            num_points=num_points,
            seed=0,
            use_all_points=use_all_points,
        ),
        "pred_label": labels[pred],
    }


def colorize_points(points):
    mins = points.min(axis=0, keepdims=True)
    maxs = points.max(axis=0, keepdims=True)
    span = np.maximum(maxs - mins, 1e-6)
    colors = (points - mins) / span
    return np.clip(colors, 0.0, 1.0)


def visualize_point_cloud(points, title):
    if o3d is None:
        raise RuntimeError("Open3D is not available in this environment.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colorize_points(points).astype(np.float64))
    o3d.visualization.draw_geometries([pcd], window_name=title)


def main():
    parser = argparse.ArgumentParser(
        description="Run PointMLP classification on a point cloud and visualize the result by default."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=str(resolve_default_pointmlp_artifact("pointmlp_best_weights.pth")),
        help="PointMLP checkpoint path.",
    )
    parser.add_argument(
        "pcd",
        nargs="?",
        default=str(TSDF_ROOT / "3d_construction" / "outputs" / "chair.pcd"),
        help="Point cloud file to classify and visualize.",
    )
    parser.add_argument(
        "--labels",
        default=str(resolve_default_pointmlp_artifact("labels.txt")),
        help="Class label file.",
    )
    parser.add_argument("--num-points", type=int, default=None, help="Number of sampled points. Defaults to the checkpoint setting.")
    parser.add_argument("--num-votes", type=int, default=1, help="Number of prediction votes.")
    parser.add_argument("--use-all-points", action="store_true", help="Use all points instead of random sampling.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device, for example cuda or cpu.",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Only print the prediction result without opening a visualization window.",
    )
    parser.add_argument(
        "--visualize-raw-points",
        action="store_true",
        help="Visualize the raw point cloud instead of the normalized input points.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    labels = load_labels(args.labels)

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    except Exception:
        ckpt = None

    model_type = ckpt.get("model_type", "pointmlp") if isinstance(ckpt, dict) else "pointmlp"
    ckpt_num_points = ckpt.get("num_points", 1024) if isinstance(ckpt, dict) else 1024
    num_points = args.num_points or ckpt_num_points

    model = PointMLPCls(k=len(labels), num_points=num_points, model_type=model_type).to(args.device)
    load_checkpoint(model, args.checkpoint, args.device)
    model.eval()

    points = load_point_cloud_points(args.pcd)
    print(f"checkpoint: {Path(args.checkpoint).resolve()}")
    print(f"point_cloud: {Path(args.pcd).resolve()}")

    result = inspect_point_cloud(
        model=model,
        points=points,
        labels=labels,
        num_points=num_points,
        device=args.device,
        num_votes=args.num_votes,
        use_all_points=args.use_all_points,
    )

    if not args.no_visualize:
        points_to_show = result["raw_points"] if args.visualize_raw_points else result["prepared_points"]
        title = f"PointMLP | pred={result['pred_label']}"
        visualize_point_cloud(points_to_show, title)


if __name__ == "__main__":
    main()
