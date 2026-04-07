import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

'''
python3 detection/pointnet2/validate_seg.py \
  --pcd 3d_construction/outputs/Area_1_office_20.pcd \
  --visualize

'''
REPO_ROOT = Path(__file__).resolve().parents[3]
TSDF_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSDF.detection.pointnet2.pointnet2seg import PointNet2SemSegSSG, SEG_INPUT_CHANNELS


DEFAULT_CHECKPOINT = Path("seg_model") / "pointnet2" / "pointnet2_seg_best.pth"
DEFAULT_LABELS_PATH = Path("seg_model") / "pointnet2" / "labels.txt"
DEFAULT_OUTPUT_DIR = Path("seg_model") / "pointnet2" / "inference_outputs"


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


def main():
    parser = argparse.ArgumentParser(
        description="Run PointNet++ semantic segmentation on a scene point cloud."
    )
    parser.add_argument("--pcd", required=True, help="Scene point cloud path (.pcd, .ply, ...).")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="PointNet++ segmentation checkpoint path. Defaults to seg_model/pointnet2/pointnet2_seg_best.pth.",
    )
    parser.add_argument(
        "--labels",
        default=str(DEFAULT_LABELS_PATH),
        help="Label file used to name predicted classes. Defaults to seg_model/pointnet2/labels.txt.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write segmentation outputs. Defaults to seg_model/pointnet2/inference_outputs.",
    )
    parser.add_argument("--target-class", default=None, help="Optional semantic class name to extract.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--max-points", type=int, default=120000)
    parser.add_argument("--min-votes", type=int, default=1)
    parser.add_argument("--max-chunks", type=int, default=256)
    parser.add_argument("--visualize", action="store_true", help="Visualize the segmented scene with Open3D.")
    args = parser.parse_args()

    checkpoint_path = resolve_repo_path(args.checkpoint)
    labels_path = resolve_repo_path(args.labels)
    output_dir = resolve_repo_path(args.output_dir)

    ckpt = read_checkpoint_metadata(checkpoint_path, device="cpu")
    num_classes = args.num_classes or ckpt.get("num_classes", 13)
    ensure_checkpoint_compatible(ckpt)
    num_points = args.num_points or ckpt.get("num_points", 4096)

    model = PointNet2SemSegSSG(num_classes=num_classes).to(args.device)
    load_checkpoint(model, checkpoint_path, args.device)
    labels = load_labels(labels_path, num_classes)

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

    output_dir.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()
