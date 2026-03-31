import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import open3d as o3d
except Exception:
    o3d = None


REPO_ROOT = Path(__file__).resolve().parents[3]
TSDF_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSDF.dataset.scanobjectnn_data import SCANOBJECTNN_LABELS, ScanObjectNNDataset
from TSDF.detection.pointmlp.pointmlp_cls import PointMLPCls


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    sampled = points[indices].astype(np.float32)
    centroid = sampled.mean(axis=0, keepdims=True)
    sampled = sampled - centroid
    scale = np.linalg.norm(sampled, axis=1).max()
    if scale > 0:
        sampled = sampled / scale
    return sampled


def prepare_points(points, num_points, seed, use_all_points=False):
    if use_all_points:
        sampled = points.astype(np.float32)
        centroid = sampled.mean(axis=0, keepdims=True)
        sampled = sampled - centroid
        scale = np.linalg.norm(sampled, axis=1).max()
        if scale > 0:
            sampled = sampled / scale
        return sampled
    return sample_points(points, num_points, seed)


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


def colorize_points(points):
    mins = points.min(axis=0, keepdims=True)
    maxs = points.max(axis=0, keepdims=True)
    span = np.maximum(maxs - mins, 1e-6)
    colors = (points - mins) / span
    return np.clip(colors, 0.0, 1.0)


def visualize_point_cloud(points, title):
    if o3d is None:
        raise RuntimeError(
            "Open3D is not available in this environment, so visualization cannot be shown."
        )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colorize_points(points).astype(np.float64))
    o3d.visualization.draw_geometries([pcd], window_name=title)


def inspect_one_sample(model, dataset, labels, num_points, device, index, num_votes, use_all_points=False, topk=3):
    raw_points = dataset.data[index][:, :3]
    target = int(dataset.labels[index])
    pred, probs = predict_with_votes(
        model, raw_points, device, num_points=num_points, num_votes=num_votes, use_all_points=use_all_points
    )

    topk = min(topk, len(labels))
    top_probs, top_indices = torch.topk(probs, k=topk)
    print(f"sample_index: {index}")
    print(f"raw_num_points: {len(raw_points)}")
    print(f"evaluation_mode: {'all_points' if use_all_points else f'{num_votes}_vote_sampling'}")
    print(f"ground_truth: {labels[target]}")
    print(f"predicted: {labels[pred]}")
    print(f"confidence: {float(top_probs[0]):.4f}")
    print("top_predictions:")
    for rank in range(topk):
        cls_idx = int(top_indices[rank].item())
        score = float(top_probs[rank].item())
        print(f"  {rank + 1}. {labels[cls_idx]} ({score:.4f})")
    return {
        "raw_points": raw_points,
        "prepared_points": prepare_points(
            raw_points,
            num_points=num_points,
            seed=0,
            use_all_points=use_all_points,
        ),
        "target_label": labels[target],
        "pred_label": labels[pred],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inspect one PointMLP prediction on ScanObjectNN."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(TSDF_ROOT / "model" / "pointmlp" / "pointmlp_best.pth"),
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--scanobjectnn-root",
        default=str(TSDF_ROOT / "data" / "ScanObjectNN"),
        help="Root directory created by download_scanobjectnn.py",
    )
    parser.add_argument(
        "--scanobjectnn-variant",
        default="pb_t50_rs",
        help="Variant for ScanObjectNN: pb_t50_rs, pb_t50_r, pb_t25, pb_t25_r, obj_bg, obj_only",
    )
    parser.add_argument("--scanobjectnn-no-bg", action="store_true")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--num-votes", type=int, default=1)
    parser.add_argument("--use-all-points", action="store_true")
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the inspected point cloud with Open3D.",
    )
    parser.add_argument(
        "--visualize-raw-points",
        action="store_true",
        help="When visualizing, show the raw point cloud instead of the normalized/evaluated points.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    labels = SCANOBJECTNN_LABELS
    ckpt = None
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
    checkpoint = load_checkpoint(model, args.checkpoint, args.device)
    model.eval()
    dataset = ScanObjectNNDataset(
        root=args.scanobjectnn_root,
        split=args.split,
        variant=args.scanobjectnn_variant,
        num_points=num_points,
        use_background=not args.scanobjectnn_no_bg,
        normalize=True,
        augment=False,
        seed=args.seed,
    )

    if args.index is None:
        args.index = random.randrange(len(dataset))
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"index {args.index} out of range for dataset of size {len(dataset)}")

    result = inspect_one_sample(
        model,
        dataset,
        labels,
        num_points,
        args.device,
        index=args.index,
        num_votes=args.num_votes,
        use_all_points=args.use_all_points,
    )

    if args.visualize:
        points_to_show = (
            result["raw_points"] if args.visualize_raw_points else result["prepared_points"]
        )
        title = f"PointMLP | gt={result['target_label']} | pred={result['pred_label']}"
        visualize_point_cloud(points_to_show, title)


if __name__ == "__main__":
    main()
