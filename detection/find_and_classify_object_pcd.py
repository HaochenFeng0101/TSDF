import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TSDF_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSDF.detection.pointnet_model import PointNetCls


def load_labels(labels_path):
    with open(labels_path, "r", encoding="utf-8") as handle:
        labels = [line.strip() for line in handle if line.strip()]
    if not labels:
        raise ValueError(f"No labels found in {labels_path}")
    return labels


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
    return missing, unexpected


def sample_and_normalize(points, num_points, rng):
    if len(points) == 0:
        raise ValueError("Cannot classify an empty point set.")

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


def classify_cluster(model, cluster_points, labels, num_points, device, rng):
    sampled = sample_and_normalize(cluster_points, num_points, rng)
    tensor = torch.from_numpy(sampled.T).unsqueeze(0).to(device=device)
    with torch.no_grad():
        logits, _ = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
    return {
        "label": labels[pred_idx],
        "confidence": float(probs[pred_idx].item()),
        "sampled_points": sampled,
    }


def remove_dominant_plane(pcd, distance_threshold, ransac_n, num_iterations):
    if len(pcd.points) < ransac_n:
        return pcd
    _, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    if not inliers:
        return pcd
    return pcd.select_by_index(inliers, invert=True)


def cluster_scene(pcd, eps, min_points, min_cluster_points):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    clusters = []
    if labels.size == 0:
        return clusters

    max_label = labels.max()
    for cluster_id in range(max_label + 1):
        indices = np.where(labels == cluster_id)[0]
        if len(indices) < min_cluster_points:
            continue
        cluster = pcd.select_by_index(indices.tolist())
        clusters.append((cluster_id, indices, cluster))
    return clusters


def write_cluster_output(cluster, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), cluster)


def main():
    parser = argparse.ArgumentParser(
        description="Cluster a scene point cloud, classify each cluster with PointNet, and search for a queried object."
    )
    parser.add_argument("--pcd", required=True, help="Scene point cloud path.")
    parser.add_argument("--checkpoint", required=True, help="PointNet checkpoint path.")
    parser.add_argument("--labels", required=True, help="Label file, one class name per line.")
    parser.add_argument("--query", required=True, help="Class name to search for.")
    parser.add_argument(
        "--output-dir",
        default=str(TSDF_ROOT / "detection" / "search_outputs"),
        help="Directory to save the matched object point cloud and metadata.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--min-points", type=int, default=30)
    parser.add_argument("--min-cluster-points", type=int, default=200)
    parser.add_argument("--remove-plane", action="store_true")
    parser.add_argument("--plane-distance-threshold", type=float, default=0.015)
    parser.add_argument("--plane-ransac-n", type=int, default=3)
    parser.add_argument("--plane-num-iterations", type=int, default=1000)
    parser.add_argument("--voxel-downsample", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    model = PointNetCls(k=len(labels)).to(args.device)
    missing, unexpected = load_checkpoint(model, args.checkpoint, args.device)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    if missing:
        raise RuntimeError(f"Missing checkpoint keys: {missing}")
    model.eval()

    pcd = o3d.io.read_point_cloud(args.pcd)
    if pcd.is_empty():
        print("not found")
        return

    if args.voxel_downsample > 0:
        pcd = pcd.voxel_down_sample(args.voxel_downsample)
    if args.remove_plane:
        pcd = remove_dominant_plane(
            pcd,
            args.plane_distance_threshold,
            args.plane_ransac_n,
            args.plane_num_iterations,
        )
    if pcd.is_empty():
        print("not found")
        return

    clusters = cluster_scene(
        pcd,
        eps=args.eps,
        min_points=args.min_points,
        min_cluster_points=args.min_cluster_points,
    )
    if not clusters:
        print("not found")
        return

    rng = np.random.default_rng(args.seed)
    predictions = []
    query = args.query.strip().lower()
    for cluster_id, indices, cluster in clusters:
        cluster_points = np.asarray(cluster.points)
        result = classify_cluster(
            model,
            cluster_points,
            labels,
            args.num_points,
            args.device,
            rng,
        )
        result.update(
            {
                "cluster_id": int(cluster_id),
                "num_points": int(len(indices)),
                "indices": indices.tolist(),
                "cluster": cluster,
            }
        )
        predictions.append(result)

    matches = [pred for pred in predictions if pred["label"].lower() == query]
    if not matches:
        print("not found")
        return

    best_match = max(matches, key=lambda item: item["confidence"])
    output_dir = Path(args.output_dir)
    object_path = output_dir / f"{query}_cluster_{best_match['cluster_id']}.pcd"
    write_cluster_output(best_match["cluster"], object_path)

    metadata = {
        "query": args.query,
        "matched_label": best_match["label"],
        "confidence": best_match["confidence"],
        "cluster_id": best_match["cluster_id"],
        "num_points": best_match["num_points"],
        "pcd": str(object_path),
        "all_predictions": [
            {
                "cluster_id": pred["cluster_id"],
                "label": pred["label"],
                "confidence": pred["confidence"],
                "num_points": pred["num_points"],
            }
            for pred in predictions
        ],
    }
    metadata_path = output_dir / f"{query}_result.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"found: {best_match['label']} ({best_match['confidence']:.4f})")
    print(f"saved object pcd: {object_path}")
    print(f"saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
