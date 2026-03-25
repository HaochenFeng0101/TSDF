import argparse
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import open3d as o3d
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TSDF_ROOT = Path(__file__).resolve().parents[1]
RANDLA_ROOT = TSDF_ROOT / "RandLA-Net-pytorch"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RANDLA_ROOT) not in sys.path:
    sys.path.insert(0, str(RANDLA_ROOT))


def install_knn_fallback():
    if "torch_points" in sys.modules or "torch_points_kernels" in sys.modules:
        return

    fallback = types.ModuleType("torch_points_kernels")

    def knn(support_points, query_points, k):
        support = support_points.float()
        query = query_points.float()
        distances = torch.cdist(query, support, p=2)
        knn_dist, knn_idx = torch.topk(distances, k=k, dim=-1, largest=False)
        return knn_idx.long(), knn_dist

    fallback.knn = knn
    sys.modules["torch_points_kernels"] = fallback


install_knn_fallback()

from model import RandLANet  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RandLA-Net semantic segmentation on a single scene point cloud and extract one target class."
    )
    parser.add_argument("--pcd", required=True, help="Scene point cloud path (.pcd, .ply, ...).")
    parser.add_argument("--checkpoint", help="RandLA-Net checkpoint path.")
    parser.add_argument(
        "--label-map",
        default=str(TSDF_ROOT / "detection" / "label_maps" / "s3dis_labels.txt"),
        help="TXT or JSON label map used to name predicted classes.",
    )
    parser.add_argument("--target-class", help="Semantic class name to extract, for example chair.")
    parser.add_argument(
        "--output-dir",
        default=str(TSDF_ROOT / "detection" / "randla_outputs"),
        help="Directory to write colored segmentation outputs and extracted objects.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-classes", type=int, default=13)
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--num-neighbors", type=int, default=16)
    parser.add_argument("--decimation", type=int, default=4)
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--max-points", type=int, default=120000)
    parser.add_argument("--min-votes", type=int, default=1)
    parser.add_argument("--max-chunks", type=int, default=256)
    parser.add_argument("--plane-distance-threshold", type=float, default=0.0)
    parser.add_argument("--dbscan-eps", type=float, default=0.06)
    parser.add_argument("--dbscan-min-points", type=int, default=30)
    parser.add_argument("--min-cluster-points", type=int, default=200)
    parser.add_argument("--random-init", action="store_true", help="Run with random weights for a smoke test.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_label_map(path, num_classes):
    map_path = Path(path)
    if not map_path.exists():
        return [f"class_{idx}" for idx in range(num_classes)]

    if map_path.suffix.lower() == ".json":
        with map_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            labels = [str(item) for item in payload]
        elif isinstance(payload, dict):
            if all(str(key).isdigit() for key in payload):
                labels = [payload[str(idx)] for idx in range(len(payload))]
            else:
                reverse = {int(value): key for key, value in payload.items()}
                labels = [reverse[idx] for idx in range(len(reverse))]
        else:
            raise ValueError(f"Unsupported JSON label map format: {map_path}")
    else:
        with map_path.open("r", encoding="utf-8") as handle:
            labels = [line.strip() for line in handle if line.strip()]

    if len(labels) < num_classes:
        labels.extend(f"class_{idx}" for idx in range(len(labels), num_classes))
    return labels[:num_classes]


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

    cleaned = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    if missing:
        raise RuntimeError(f"Missing checkpoint keys: {missing}")


def load_scene(path):
    scene = o3d.io.read_point_cloud(str(path))
    if scene.is_empty():
        raise ValueError(f"Failed to read a non-empty point cloud from {path}")
    return scene


def remove_plane(scene, distance_threshold):
    if distance_threshold <= 0 or len(scene.points) < 3:
        return scene
    _, inliers = scene.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=1000,
    )
    if not inliers:
        return scene
    return scene.select_by_index(inliers, invert=True)


def preprocess_scene(scene, voxel_size, max_points):
    current = scene
    if voxel_size > 0:
        current = current.voxel_down_sample(voxel_size)

    if max_points > 0 and len(current.points) > max_points:
        ratio = len(current.points) / float(max_points)
        sampled = math.ceil(ratio)
        current = current.uniform_down_sample(sampled)

    points = np.asarray(current.points, dtype=np.float32)
    if current.has_colors():
        colors = np.asarray(current.colors, dtype=np.float32)
    else:
        colors = np.zeros((len(points), 3), dtype=np.float32)

    return current, points, colors


def choose_chunk_indices(points, num_points, min_votes, max_chunks):
    num_scene_points = len(points)
    if num_scene_points == 0:
        return []

    if num_scene_points <= num_points:
        indices = np.arange(num_scene_points, dtype=np.int64)
        return [indices]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    coverage = np.zeros(num_scene_points, dtype=np.int32)
    chunk_indices = []
    cursor = 0

    while coverage.min() < min_votes and len(chunk_indices) < max_chunks:
        center_idx = int(np.argmin(coverage))
        center = points[center_idx]
        _, idx, _ = kdtree.search_knn_vector_3d(center.astype(np.float64), min(num_points, num_scene_points))
        chunk = np.asarray(idx, dtype=np.int64)
        if chunk.size < num_points:
            pad = np.random.default_rng(cursor).choice(chunk, num_points - chunk.size, replace=True)
            chunk = np.concatenate([chunk, pad])
        coverage[chunk] += 1
        chunk_indices.append(chunk)
        cursor += 1

    return chunk_indices


def build_chunk_features(points, colors, indices):
    chunk_points = points[indices]
    chunk_colors = colors[indices]
    center = chunk_points.mean(axis=0, keepdims=True)
    centered_points = chunk_points - center
    return np.concatenate([centered_points, chunk_colors], axis=1).astype(np.float32)


def predict_scene(model, points, colors, num_points, min_votes, max_chunks, device):
    chunks = choose_chunk_indices(points, num_points, min_votes, max_chunks)
    if not chunks:
        raise ValueError("No chunks were generated for scene inference.")

    num_classes = model.fc_end[-1].conv.out_channels
    logits_sum = np.zeros((len(points), num_classes), dtype=np.float32)
    logits_count = np.zeros(len(points), dtype=np.int32)

    model.eval()
    with torch.no_grad():
        for chunk_ids in chunks:
            features = build_chunk_features(points, colors, chunk_ids)
            tensor = torch.from_numpy(features).unsqueeze(0).to(device)
            scores = model(tensor).squeeze(0).transpose(0, 1).cpu().numpy()
            logits_sum[chunk_ids] += scores
            logits_count[chunk_ids] += 1

    unseen_mask = logits_count == 0
    if unseen_mask.any():
        logits_count[unseen_mask] = 1

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


def extract_target(points, colors, predictions, labels, target_label, eps, min_points, min_cluster_points, output_dir):
    normalized = target_label.strip().lower()
    label_to_index = {name.lower(): idx for idx, name in enumerate(labels)}
    if normalized not in label_to_index:
        raise ValueError(f"Target class '{target_label}' is not present in the label map.")

    target_idx = label_to_index[normalized]
    mask = predictions == target_idx
    if not np.any(mask):
        return None

    target_points = points[mask]
    target_colors = colors[mask]

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points.astype(np.float64))
    target_pcd.colors = o3d.utility.Vector3dVector(target_colors.astype(np.float64))

    merged_path = output_dir / f"{normalized}.ply"
    o3d.io.write_point_cloud(str(merged_path), target_pcd)

    clusters = []
    if len(target_points) >= min_points:
        cluster_labels = np.array(target_pcd.cluster_dbscan(eps=eps, min_points=min_points))
        if cluster_labels.size > 0 and cluster_labels.max() >= 0:
            for cluster_id in range(cluster_labels.max() + 1):
                cluster_mask = cluster_labels == cluster_id
                if int(cluster_mask.sum()) < min_cluster_points:
                    continue
                cluster_pcd = target_pcd.select_by_index(np.where(cluster_mask)[0].tolist())
                cluster_path = output_dir / f"{normalized}_cluster_{cluster_id:03d}.ply"
                o3d.io.write_point_cloud(str(cluster_path), cluster_pcd)
                clusters.append(
                    {
                        "cluster_id": int(cluster_id),
                        "num_points": int(cluster_mask.sum()),
                        "path": str(cluster_path),
                    }
                )

    return {
        "target_class": target_label,
        "target_index": int(target_idx),
        "num_points": int(mask.sum()),
        "merged_path": str(merged_path),
        "clusters": clusters,
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.random_init and not args.checkpoint:
        raise ValueError("Provide --checkpoint for real inference, or use --random-init for a smoke test.")

    labels = load_label_map(args.label_map, args.num_classes)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    scene = load_scene(args.pcd)
    scene = remove_plane(scene, args.plane_distance_threshold)
    scene, points, colors = preprocess_scene(scene, args.voxel_size, args.max_points)

    model = RandLANet(
        d_in=6,
        num_classes=args.num_classes,
        num_neighbors=args.num_neighbors,
        decimation=args.decimation,
        device=device,
    ).to(device)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device)

    predictions, logits, num_chunks = predict_scene(
        model=model,
        points=points,
        colors=colors,
        num_points=args.num_points,
        min_votes=args.min_votes,
        max_chunks=args.max_chunks,
        device=device,
    )

    segmented_path = output_dir / "scene_segmented.ply"
    write_colored_scene(segmented_path, points, colorize_predictions(predictions))
    np.save(output_dir / "scene_predictions.npy", predictions)

    class_hist = {}
    for class_idx, count in zip(*np.unique(predictions, return_counts=True)):
        class_hist[labels[int(class_idx)]] = int(count)

    metadata = {
        "source_pcd": str(Path(args.pcd).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "random_init": bool(args.random_init),
        "num_points_after_preprocess": int(len(points)),
        "num_chunks": int(num_chunks),
        "num_classes": int(args.num_classes),
        "label_map": labels,
        "class_histogram": class_hist,
        "segmented_scene": str(segmented_path),
    }

    if args.target_class:
        extracted = extract_target(
            points=points,
            colors=colors,
            predictions=predictions,
            labels=labels,
            target_label=args.target_class,
            eps=args.dbscan_eps,
            min_points=args.dbscan_min_points,
            min_cluster_points=args.min_cluster_points,
            output_dir=output_dir,
        )
        metadata["target_extraction"] = extracted

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"saved segmented scene: {segmented_path}")
    print(f"saved metadata: {metadata_path}")
    if args.target_class:
        extraction = metadata.get("target_extraction")
        if extraction is None:
            print(f"target class '{args.target_class}' was not predicted in this scene")
        else:
            print(f"saved merged target cloud: {extraction['merged_path']}")


if __name__ == "__main__":
    main()
