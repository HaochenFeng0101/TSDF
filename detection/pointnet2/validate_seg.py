import argparse
import json
import math
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import open3d as o3d
import torch

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

'''
python3 detection/pointnet2/validate_seg.py \
  --pcd 3d_construction/outputs/Area_1_office_20.pcd \
  --checkpoint /home/haochen/code/TSDF/model/pointnet2_seg/pointnet2_seg_best.pth \
  --labels /home/haochen/code/TSDF/model/pointnet2_seg/labels.txt \
  --visualize
  
  cd /home/haochen/code/TSDF
python3 detection/pointnet2/validate_seg.py \
  --pcd /home/haochen/code/TSDF/3d_construction/outputs/fr3_office.pcd \
  --checkpoint /home/haochen/code/TSDF/model/pointnet2_seg/pointnet2_seg_best.pth \
  --labels /home/haochen/code/TSDF/model/pointnet2_seg/labels.txt \
  --visualize


'''
REPO_ROOT = Path(__file__).resolve().parents[3]
TSDF_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSDF.detection.pointnet2.pointnet2seg import PointNet2SemSegSSG, SEG_INPUT_CHANNELS
from TSDF.detection.pointnet2.train_seg import load_seg_sample


DEFAULT_CHECKPOINT = Path("model") / "pointnet2_seg" / "pointnet2_seg_best.pth"
DEFAULT_LABELS_PATH = Path("model") / "pointnet2_seg" / "labels.txt"
DEFAULT_OUTPUT_DIR = Path("model") / "pointnet2_seg" / "inference_outputs"


def resolve_repo_path(path_str):
    path = Path(path_str)
    if not path.is_absolute():
        path = TSDF_ROOT / path
    return path


def load_labels(path, num_classes):
    labels_path = Path(path)
    if not labels_path.exists():
        return [f"class_{idx}" for idx in range(num_classes)]
    with open(labels_path, "r", encoding="utf-8") as handle:
        labels = [line.strip() for line in handle if line.strip()]
    if len(labels) < num_classes:
        labels.extend(f"class_{idx}" for idx in range(len(labels), num_classes))
    return labels[:num_classes]


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


def read_checkpoint_metadata(checkpoint_path, device="cpu"):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def ensure_checkpoint_compatible(checkpoint):
    input_channels = int(checkpoint.get("input_channels", SEG_INPUT_CHANNELS))
    if input_channels != SEG_INPUT_CHANNELS:
        raise RuntimeError(
            f"Expected a {SEG_INPUT_CHANNELS}-channel segmentation checkpoint, got input_channels={input_channels}."
        )


def load_scene(path):
    scene = o3d.io.read_point_cloud(str(path))
    if scene.is_empty():
        raise ValueError(f"Failed to read a non-empty point cloud from {path}")
    points = np.asarray(scene.points, dtype=np.float32)
    colors = np.asarray(scene.colors, dtype=np.float32) if scene.has_colors() else np.zeros((len(points), 3), dtype=np.float32)
    return scene, points, colors


def split_points_and_colors(points):
    points = np.asarray(points, dtype=np.float32)
    xyz = points[:, :3]
    if points.shape[1] >= 6:
        colors = points[:, 3:6].astype(np.float32).copy()
        if colors.max() > 1.0:
            colors = colors / 255.0
        colors = np.clip(colors, 0.0, 1.0)
    else:
        colors = np.zeros((len(points), 3), dtype=np.float32)
    return xyz, colors


def preprocess_scene(scene, voxel_size, max_points):
    current = scene
    if voxel_size > 0:
        current = current.voxel_down_sample(voxel_size)

    if max_points > 0 and len(current.points) > max_points:
        ratio = len(current.points) / float(max_points)
        sampled = math.ceil(ratio)
        current = current.uniform_down_sample(sampled)

    points = np.asarray(current.points, dtype=np.float32)
    colors = np.asarray(current.colors, dtype=np.float32) if current.has_colors() else np.zeros((len(points), 3), dtype=np.float32)
    return current, points, colors


def choose_chunk_indices(points, num_points, min_votes, max_chunks):
    if len(points) == 0:
        return []
    if len(points) <= num_points:
        return [np.arange(len(points), dtype=np.int64)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    coverage = np.zeros(len(points), dtype=np.int32)
    chunks = []
    cursor = 0
    while coverage.min() < min_votes and len(chunks) < max_chunks:
        center_idx = int(np.argmin(coverage))
        center = points[center_idx]
        _, idx, _ = kdtree.search_knn_vector_3d(center.astype(np.float64), min(num_points, len(points)))
        chunk = np.asarray(idx, dtype=np.int64)
        if chunk.size < num_points:
            pad = np.random.default_rng(cursor).choice(chunk, num_points - chunk.size, replace=True)
            chunk = np.concatenate([chunk, pad])
        coverage[chunk] += 1
        chunks.append(chunk)
        cursor += 1
    return chunks


def normalize_xyz(points):
    xyz = points[:, :3].astype(np.float32)
    centroid = xyz.mean(axis=0, keepdims=True)
    xyz = xyz - centroid
    scale = np.linalg.norm(xyz, axis=1).max()
    if scale > 0:
        xyz = xyz / scale
    result = points.astype(np.float32).copy()
    result[:, :3] = xyz
    return result


def compute_room_normalized_xyz(points):
    mins = points.min(axis=0, keepdims=True)
    maxs = points.max(axis=0, keepdims=True)
    span = np.maximum(maxs - mins, 1e-6)
    return (points - mins) / span


def build_chunk_features(points, colors, room_norm_xyz, indices):
    chunk_points = points[indices]
    chunk_colors = colors[indices]
    features = np.concatenate([chunk_points, chunk_colors, room_norm_xyz[indices]], axis=1)
    return normalize_xyz(features)


def predict_scene(model, points, colors, device, num_points, min_votes, max_chunks):
    chunks = choose_chunk_indices(points, num_points, min_votes, max_chunks)
    if not chunks:
        raise ValueError("No chunks were generated for scene inference.")

    logits_sum = None
    logits_count = np.zeros(len(points), dtype=np.int32)
    room_norm_xyz = compute_room_normalized_xyz(points)

    model.eval()
    with torch.no_grad():
        for chunk_ids in chunks:
            features = build_chunk_features(points, colors, room_norm_xyz, chunk_ids)
            tensor = torch.from_numpy(features.T).unsqueeze(0).to(device)
            scores = model(tensor).squeeze(0).transpose(0, 1).cpu().numpy()
            if logits_sum is None:
                logits_sum = np.zeros((len(points), scores.shape[1]), dtype=np.float32)
            logits_sum[chunk_ids] += scores
            logits_count[chunk_ids] += 1

    unseen = logits_count == 0
    logits_count[unseen] = 1
    averaged_logits = logits_sum / logits_count[:, None]
    predictions = np.argmax(averaged_logits, axis=1).astype(np.int32)
    return predictions, averaged_logits, len(chunks)


def colorize_predictions(predictions):
    palette = np.asarray(
        [
            [166, 206, 227],
            [31, 120, 180],
            [178, 223, 138],
            [51, 160, 44],
            [251, 154, 153],
            [227, 26, 28],
            [253, 191, 111],
            [255, 127, 0],
            [202, 178, 214],
            [106, 61, 154],
            [255, 255, 153],
            [177, 89, 40],
            [141, 211, 199],
            [255, 255, 179],
            [190, 186, 218],
            [251, 128, 114],
        ],
        dtype=np.float32,
    ) / 255.0
    return palette[predictions % len(palette)]


def write_colored_scene(path, points, colors):
    scene = o3d.geometry.PointCloud()
    scene.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    scene.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(str(path), scene)


def write_masked_object(path, points, colors, mask):
    selected = points[mask]
    selected_colors = colors[mask]
    if len(selected) == 0:
        return False
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(selected.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(selected_colors.astype(np.float64))
    o3d.io.write_point_cloud(str(path), cloud)
    return True


def make_run_output_dir(base_dir: Path, prefix: str):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{prefix}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def compute_dataset_iou_records(predictions, labels, num_classes):
    inter = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)
    for class_idx in range(num_classes):
        pred_mask = predictions == class_idx
        label_mask = labels == class_idx
        inter[class_idx] += np.logical_and(pred_mask, label_mask).sum()
        union[class_idx] += np.logical_or(pred_mask, label_mask).sum()
    per_class_iou = [
        (float(inter[class_idx] / union[class_idx]) if union[class_idx] > 0 else None)
        for class_idx in range(num_classes)
    ]
    valid = [value for value in per_class_iou if value is not None]
    miou = float(np.mean(valid)) if valid else 0.0
    return per_class_iou, miou


def write_dataset_iou_txt(path, label_names, per_class_iou, miou):
    percent_values = []
    lines = []
    for class_idx, label_name in enumerate(label_names):
        iou = per_class_iou[class_idx]
        percent = 0.0 if iou is None else iou * 100.0
        percent_values.append(f"{percent:.1f}")
        lines.append(f"{label_name}: {percent:.1f}")
    lines.append(f"mIoU: {miou * 100.0:.1f}")
    lines.append("")
    lines.append("table_row:")
    lines.append(" ".join(percent_values + [f"{miou * 100.0:.1f}"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Run PointNet++ semantic segmentation on a scene point cloud."
    )
    parser.add_argument("--pcd", default=None, help="Scene point cloud path (.pcd, .ply, ...).")
    parser.add_argument("--data-root", default=None, help="Optional dataset root containing split folders such as val/ with npz/npy/pcd samples.")
    parser.add_argument("--split", default="val", help="Dataset split name used with --data-root. Defaults to val.")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="PointNet++ segmentation checkpoint path. Defaults to model/pointnet2_seg/pointnet2_seg_best.pth.",
    )
    parser.add_argument(
        "--labels",
        default=str(DEFAULT_LABELS_PATH),
        help="Label file used to name predicted classes. Defaults to model/pointnet2_seg/labels.txt.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write segmentation outputs. Defaults to model/pointnet2_seg/inference_outputs.",
    )
    parser.add_argument("--target-class", default=None, help="Optional semantic class name to extract.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--max-points", type=int, default=120000)
    parser.add_argument("--min-votes", type=int, default=1)
    parser.add_argument("--max-chunks", type=int, default=256)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on the number of dataset samples to evaluate. Useful for quick checks.",
    )
    parser.add_argument("--visualize", action="store_true", help="Visualize the segmented scene with Open3D.")
    args = parser.parse_args()

    checkpoint_path = resolve_repo_path(args.checkpoint)
    labels_path = resolve_repo_path(args.labels)
    base_output_dir = resolve_repo_path(args.output_dir)

    if not args.pcd and not args.data_root:
        raise ValueError("Provide either --pcd for a single scene or --data-root for dataset evaluation.")
    if args.pcd and args.data_root:
        raise ValueError("Use either --pcd or --data-root, not both at the same time.")

    ckpt = read_checkpoint_metadata(checkpoint_path, device="cpu")
    num_classes = args.num_classes or ckpt.get("num_classes", 13)
    ensure_checkpoint_compatible(ckpt)
    num_points = args.num_points or ckpt.get("num_points", 4096)

    model = PointNet2SemSegSSG(num_classes=num_classes).to(args.device)
    load_checkpoint(model, checkpoint_path, args.device)
    labels = load_labels(labels_path, num_classes)

    if args.pcd:
        output_dir = make_run_output_dir(base_output_dir, "single_scene")
        scene, _, _ = load_scene(args.pcd)
        scene, points, colors = preprocess_scene(scene, args.voxel_size, args.max_points)
        predictions, logits, num_chunks = predict_scene(
            model,
            points,
            colors,
            args.device,
            num_points=num_points,
            min_votes=args.min_votes,
            max_chunks=args.max_chunks,
        )

        segmented_path = output_dir / "scene_segmented.ply"
        pred_path = output_dir / "scene_predictions.npy"
        meta_path = output_dir / "scene_metadata.json"
        write_colored_scene(segmented_path, points, colorize_predictions(predictions))
        np.save(pred_path, predictions)

        metadata = {
            "pcd": str(Path(args.pcd).resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "num_points_used": int(len(points)),
            "num_chunks": int(num_chunks),
            "num_classes": int(num_classes),
            "input_channels": SEG_INPUT_CHANNELS,
            "labels": labels,
            "segmented_scene": str(segmented_path),
            "predictions": str(pred_path),
        }

        if args.target_class:
            target = args.target_class.strip().lower()
            label_to_idx = {label.lower(): idx for idx, label in enumerate(labels)}
            if target not in label_to_idx:
                raise ValueError(f"Unknown target class '{args.target_class}'. Available classes: {labels}")
            target_idx = label_to_idx[target]
            mask = predictions == target_idx
            target_path = output_dir / f"{target}_points.ply"
            if write_masked_object(target_path, points, colorize_predictions(predictions), mask):
                metadata["target_class"] = args.target_class
                metadata["target_output"] = str(target_path)
                metadata["target_points"] = int(mask.sum())

        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        print(f"saved segmented scene: {segmented_path}")
        print(f"saved predictions: {pred_path}")
        print(f"saved metadata: {meta_path}")

        if args.visualize:
            colored = o3d.geometry.PointCloud()
            colored.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            colored.colors = o3d.utility.Vector3dVector(colorize_predictions(predictions).astype(np.float64))
            o3d.visualization.draw_geometries([colored], window_name="PointNet++ Segmentation")
    else:
        data_root = resolve_repo_path(args.data_root)
        split_dir = data_root / args.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")

        files = sorted(
            path for path in split_dir.iterdir() if path.suffix.lower() in {".npz", ".npy", ".pcd", ".ply"}
        )
        if not files:
            raise RuntimeError(f"No dataset samples found under {split_dir}")
        if args.max_samples is not None and args.max_samples > 0:
            files = files[: args.max_samples]

        output_dir = make_run_output_dir(base_output_dir, f"dataset_{args.split}")
        per_file = []
        total_correct = 0
        total_seen = 0
        total_inter = np.zeros(num_classes, dtype=np.float64)
        total_union = np.zeros(num_classes, dtype=np.float64)

        iterator = files
        if tqdm is not None:
            iterator = tqdm(files, desc=f"dataset_{args.split}", dynamic_ncols=True)

        for path in iterator:
            raw_points, gt_labels = load_seg_sample(path)
            points, colors = split_points_and_colors(raw_points)
            if args.max_points > 0 and len(points) > args.max_points:
                ratio = len(points) / float(args.max_points)
                sampled = math.ceil(ratio)
                points = points[::sampled]
                colors = colors[::sampled]
                gt_labels = gt_labels[::sampled]

            predictions, _, num_chunks = predict_scene(
                model,
                points,
                colors,
                args.device,
                num_points=num_points,
                min_votes=args.min_votes,
                max_chunks=args.max_chunks,
            )

            total_correct += int((predictions == gt_labels).sum())
            total_seen += int(len(gt_labels))
            sample_iou, sample_miou = compute_dataset_iou_records(predictions, gt_labels, num_classes)

            for class_idx in range(num_classes):
                pred_mask = predictions == class_idx
                label_mask = gt_labels == class_idx
                total_inter[class_idx] += np.logical_and(pred_mask, label_mask).sum()
                total_union[class_idx] += np.logical_or(pred_mask, label_mask).sum()

            per_file.append(
                {
                    "file": str(path.resolve()),
                    "num_points": int(len(points)),
                    "num_chunks": int(num_chunks),
                    "miou": sample_miou,
                    "per_class_iou": {
                        labels[class_idx]: sample_iou[class_idx] for class_idx in range(num_classes)
                    },
                }
            )
            if tqdm is not None:
                current_acc = total_correct / max(total_seen, 1)
                current_valid_ious = [
                    (total_inter[class_idx] / total_union[class_idx])
                    for class_idx in range(num_classes)
                    if total_union[class_idx] > 0
                ]
                current_miou = float(np.mean(current_valid_ious)) if current_valid_ious else 0.0
                iterator.set_postfix(acc=f"{current_acc:.4f}", miou=f"{current_miou:.4f}")

        per_class_iou = [
            (float(total_inter[class_idx] / total_union[class_idx]) if total_union[class_idx] > 0 else None)
            for class_idx in range(num_classes)
        ]
        miou = float(np.mean([value for value in per_class_iou if value is not None])) if any(
            value is not None for value in per_class_iou
        ) else 0.0
        acc = float(total_correct / max(total_seen, 1))

        meta_path = output_dir / "dataset_metrics.json"
        txt_path = output_dir / "iou_percent.txt"
        payload = {
            "data_root": str(data_root.resolve()),
            "split": args.split,
            "checkpoint": str(checkpoint_path.resolve()),
            "labels": labels,
            "accuracy": acc,
            "miou": miou,
            "per_class_iou": {labels[class_idx]: per_class_iou[class_idx] for class_idx in range(num_classes)},
            "files": per_file,
        }
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        write_dataset_iou_txt(txt_path, labels, per_class_iou, miou)

        print(f"saved dataset metrics: {meta_path}")
        print(f"saved IoU txt: {txt_path}")
        print(f"dataset_acc={acc:.4f} | dataset_mIoU={miou:.4f}")
        print("per_class_iou_percent | " + " ".join(
            f"{0.0 if value is None else value * 100.0:.1f}" for value in per_class_iou
        ))


if __name__ == "__main__":
    main()
