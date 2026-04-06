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

from TSDF.dataset.extra_object_data import (
    ModelNet40H5Dataset,
    ModelNet40OffDataset,
    load_off_mesh,
    load_point_file,
    modelnet40_root_exists,
)
from TSDF.dataset.scanobjectnn_data import SCANOBJECTNN_LABELS, ScanObjectNNDataset
from TSDF.detection.pointnet.pointnet_cls import PointNetCls


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scanobjectnn_is_ready(root):
    root = Path(root)
    return root.exists() and ((root / "h5_files" / "main_split").exists() or (root / "main_split").exists())


def resolve_dataset_type(args):
    if args.dataset_type != "auto":
        return args.dataset_type
    if scanobjectnn_is_ready(args.scanobjectnn_root):
        return "scanobjectnn"
    if modelnet40_root_exists(args.modelnet40_root):
        return "modelnet40"
    raise RuntimeError("No usable dataset found for dataset-type auto.")


def load_labels(labels_path):
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find label file: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        labels = [line.strip() for line in handle if line.strip()]
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels


def load_checkpoint_file(checkpoint_path, map_location):
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = load_checkpoint_file(checkpoint_path, device)
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


def infer_num_classes_from_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        return None
    labels = checkpoint.get("labels")
    if isinstance(labels, list) and labels:
        return len(labels)
    return None


def build_modelnet_dataset(args, num_points, seed):
    modelnet_root = Path(args.modelnet40_root)
    h5_ready = (
        (modelnet_root / "modelnet40_ply_hdf5_2048" / "train_files.txt").exists()
        or (modelnet_root / "train_files.txt").exists()
    )
    if h5_ready:
        dataset = ModelNet40H5Dataset(
            root=args.modelnet40_root,
            split=args.split,
            num_points=num_points,
            augment=False,
            seed=seed if seed is not None else 0,
        )
    else:
        dataset = ModelNet40OffDataset(
            root=args.modelnet40_root,
            split=args.split,
            num_points=num_points,
            augment=False,
            seed=seed if seed is not None else 0,
        )
    return dataset


def sample_points(points, num_points, seed):
    rng = np.random.default_rng(seed)
    if len(points) >= num_points:
        indices = rng.choice(len(points), num_points, replace=False)
    else:
        indices = rng.choice(len(points), num_points, replace=True)
    return points[indices].astype(np.float32)


def normalize_points(points):
    points = points.astype(np.float32)
    centroid = points.mean(axis=0, keepdims=True)
    points = points - centroid
    scale = np.linalg.norm(points, axis=1).max()
    if scale > 0:
        points = points / scale
    return points


def prepare_points(points, num_points, seed, use_all_points=False):
    sampled = points.astype(np.float32) if use_all_points else sample_points(points, num_points, seed)
    return normalize_points(sampled)


def predict_with_votes(model, points, device, num_points, num_votes, use_all_points=False):
    logits_votes = []
    total_votes = 1 if use_all_points else max(num_votes, 1)
    for vote_idx in range(total_votes):
        prepared = prepare_points(points, num_points=num_points, seed=vote_idx, use_all_points=use_all_points)
        tensor = torch.from_numpy(prepared.T).unsqueeze(0).to(device=device)
        with torch.no_grad():
            logits, _ = model(tensor)
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
        raise RuntimeError("Open3D is not available in this environment.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colorize_points(points).astype(np.float64))
    o3d.visualization.draw_geometries([pcd], window_name=title)


def load_point_cloud_points(point_cloud_path):
    path = Path(point_cloud_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find point cloud file: {path}")
    points = load_point_file(path)
    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"Expected Nx3 points in {path}, got shape {points.shape}")
    return points


def get_dataset_target(dataset, index):
    if isinstance(dataset, ScanObjectNNDataset):
        return int(dataset.labels[index])
    if hasattr(dataset, "samples"):
        sample = dataset.samples[index]
        if isinstance(sample, dict) and "label_idx" in sample:
            return int(sample["label_idx"])
    raise ValueError(f"Cannot read target label from dataset type: {type(dataset).__name__}")


def get_dataset_raw_points(dataset, index):
    if isinstance(dataset, ScanObjectNNDataset):
        return dataset.data[index][:, :3].astype(np.float32)

    if isinstance(dataset, ModelNet40H5Dataset):
        return np.asarray(dataset.samples[index]["points"], dtype=np.float32)[:, :3]

    if isinstance(dataset, ModelNet40OffDataset):
        sample_path = Path(dataset.samples[index]["path"])
        if sample_path.suffix.lower() == ".off":
            vertices, _ = load_off_mesh(sample_path)
            return vertices[np.isfinite(vertices).all(axis=1)].astype(np.float32)
        return load_point_file(sample_path).astype(np.float32)

    raise ValueError(f"Cannot extract raw points from dataset type: {type(dataset).__name__}")


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
        "prepared_points": prepare_points(points, num_points=num_points, seed=0, use_all_points=use_all_points),
        "pred_label": labels[pred],
    }


def inspect_one_sample(model, dataset, labels, num_points, device, index, num_votes, use_all_points=False, topk=3):
    raw_points = get_dataset_raw_points(dataset, index)
    target = get_dataset_target(dataset, index)
    pred, probs = predict_with_votes(
        model=model,
        points=raw_points,
        device=device,
        num_points=num_points,
        num_votes=num_votes,
        use_all_points=use_all_points,
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
        "prepared_points": prepare_points(raw_points, num_points=num_points, seed=0, use_all_points=use_all_points),
        "target_label": labels[target],
        "pred_label": labels[pred],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Inspect one PointNet prediction on ScanObjectNN/ModelNet40 or an external point cloud."
    )
    parser.add_argument("--checkpoint", default=str(TSDF_ROOT / "model" / "pointnet" / "pointnet_best.pth"))
    parser.add_argument("--labels", default=str(TSDF_ROOT / "model" / "pointnet" / "labels.txt"))
    parser.add_argument("--point-cloud", default=None)
    parser.add_argument(
        "--dataset-type",
        choices=["auto", "scanobjectnn", "modelnet40"],
        default="auto",
        help="auto prefers ScanObjectNN and falls back to ModelNet40.",
    )
    parser.add_argument("--scanobjectnn-root", default=str(TSDF_ROOT / "data" / "ScanObjectNN"))
    parser.add_argument("--scanobjectnn-variant", default="pb_t50_rs")
    parser.add_argument("--scanobjectnn-no-bg", action="store_true")
    parser.add_argument("--modelnet40-root", default=str(TSDF_ROOT / "data" / "ModelNet40"))
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--num-votes", type=int, default=1)
    parser.add_argument("--use-all-points", action="store_true")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize-raw-points", action="store_true")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    checkpoint = load_checkpoint_file(args.checkpoint, "cpu")
    ckpt_labels = checkpoint.get("labels") if isinstance(checkpoint, dict) else None
    ckpt_num_points = checkpoint.get("num_points", 1024) if isinstance(checkpoint, dict) else 1024
    ckpt_num_classes = infer_num_classes_from_checkpoint(checkpoint)

    num_points = args.num_points or ckpt_num_points

    dataset = None
    dataset_type = None
    if args.point_cloud is None:
        dataset_type = resolve_dataset_type(args)
        if dataset_type == "scanobjectnn":
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
            default_labels = list(SCANOBJECTNN_LABELS)
        else:
            dataset = build_modelnet_dataset(args, num_points=num_points, seed=args.seed)
            default_labels = list(dataset.labels)
    else:
        default_labels = []

    if args.labels and Path(args.labels).exists():
        labels = load_labels(args.labels)
    elif isinstance(ckpt_labels, list) and ckpt_labels:
        labels = ckpt_labels
    elif default_labels:
        labels = default_labels
    else:
        raise RuntimeError("Could not resolve labels. Provide --labels or use a checkpoint containing labels.")

    if ckpt_num_classes is not None and ckpt_num_classes != len(labels):
        raise RuntimeError(
            f"Label count ({len(labels)}) does not match checkpoint classes ({ckpt_num_classes})."
        )

    dropout = checkpoint.get("dropout", 0.4) if isinstance(checkpoint, dict) else 0.4
    model = PointNetCls(k=len(labels), dropout=dropout).to(args.device)
    load_checkpoint(model, args.checkpoint, args.device)
    model.eval()

    if args.point_cloud:
        points = load_point_cloud_points(args.point_cloud)
        print(f"point_cloud: {Path(args.point_cloud).resolve()}")
        result = inspect_point_cloud(
            model=model,
            points=points,
            labels=labels,
            num_points=num_points,
            device=args.device,
            num_votes=args.num_votes,
            use_all_points=args.use_all_points,
            topk=args.topk,
        )
    else:
        if args.index is None:
            args.index = random.randrange(len(dataset))
        if args.index < 0 or args.index >= len(dataset):
            raise IndexError(f"index {args.index} out of range for dataset of size {len(dataset)}")
        print(f"dataset_type: {dataset_type}")
        result = inspect_one_sample(
            model=model,
            dataset=dataset,
            labels=labels,
            num_points=num_points,
            device=args.device,
            index=args.index,
            num_votes=args.num_votes,
            use_all_points=args.use_all_points,
            topk=args.topk,
        )

    if args.visualize:
        points_to_show = result["raw_points"] if args.visualize_raw_points else result["prepared_points"]
        title = f"PointNet | pred={result['pred_label']}"
        if "target_label" in result:
            title = f"PointNet | gt={result['target_label']} | pred={result['pred_label']}"
        visualize_point_cloud(points_to_show, title)


if __name__ == "__main__":
    main()
