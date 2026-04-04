import argparse
import json
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

from TSDF.detection.pointmlp.pointmlp_cls import PointMLPCls


def load_labels(labels_path):
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        labels = [line.strip() for line in handle if line.strip()]
    if not labels:
        raise ValueError(f"Label file is empty: {path}")
    return labels


def load_raw_checkpoint(checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def clean_state_dict(checkpoint):
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = load_raw_checkpoint(checkpoint_path, device)
    cleaned = clean_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        raise RuntimeError(f"Missing checkpoint keys: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    return checkpoint


def load_point_cloud_points(path):
    if o3d is None:
        raise RuntimeError("Open3D is required to read .pcd files.")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {path}")

    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise RuntimeError(f"Point cloud is empty or unreadable: {path}")

    points = np.asarray(cloud.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise RuntimeError(f"Expected Nx3 points in {path}, got shape {points.shape}")
    return points[:, :3]


def normalize_points(points):
    points = points.astype(np.float32)
    centroid = points.mean(axis=0, keepdims=True)
    points = points - centroid
    scale = np.linalg.norm(points, axis=1).max()
    if scale > 0:
        points = points / scale
    return points


def sample_points(points, num_points, seed):
    rng = np.random.default_rng(seed)
    if len(points) >= num_points:
        indices = rng.choice(len(points), num_points, replace=False)
    else:
        indices = rng.choice(len(points), num_points, replace=True)
    return points[indices].astype(np.float32)


def prepare_points(points, num_points, seed, use_all_points=False):
    if use_all_points:
        return normalize_points(points)
    return normalize_points(sample_points(points, num_points, seed))


def predict_with_votes(model, points, device, num_points, num_votes, seed, use_all_points=False):
    logits_votes = []
    total_votes = 1 if use_all_points else max(num_votes, 1)
    for vote_idx in range(total_votes):
        prepared = prepare_points(
            points,
            num_points=num_points,
            seed=seed + vote_idx,
            use_all_points=use_all_points,
        )
        tensor = torch.from_numpy(prepared.T).unsqueeze(0).to(device=device)
        with torch.no_grad():
            logits = model(tensor)
        logits_votes.append(logits[0].detach().cpu())

    mean_logits = torch.stack(logits_votes, dim=0).mean(dim=0)
    probs = torch.softmax(mean_logits, dim=0)
    pred_idx = int(torch.argmax(probs).item())
    return pred_idx, probs


def collect_category_files(input_root):
    input_root = Path(input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    category_to_files = {}
    for category_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        files = sorted(category_dir.rglob("*.pcd"))
        category_to_files[category_dir.name] = files
    return category_to_files


def build_result_record(file_path, source_category, raw_points, labels, pred_idx, probs, topk):
    topk = min(topk, len(labels))
    top_probs, top_indices = torch.topk(probs, k=topk)

    score_map = {
        label: float(probs[idx].item())
        for idx, label in enumerate(labels)
    }
    top_predictions = []
    for rank in range(topk):
        cls_idx = int(top_indices[rank].item())
        top_predictions.append(
            {
                "rank": rank + 1,
                "label": labels[cls_idx],
                "score": float(top_probs[rank].item()),
                "class_index": cls_idx,
            }
        )

    return {
        "file_name": file_path.name,
        "file_path": str(file_path.resolve()),
        "source_category": source_category,
        "source_category_in_label_space": source_category in score_map,
        "source_category_score": float(score_map[source_category]) if source_category in score_map else None,
        "raw_num_points": int(len(raw_points)),
        "predicted_label": labels[pred_idx],
        "predicted_index": pred_idx,
        "confidence": float(probs[pred_idx].item()),
        "scores": score_map,
        "top_predictions": top_predictions,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Batch run PointMLP on all .pcd files under data/extra_object and save scores to JSON."
    )
    parser.add_argument(
        "--input-root",
        default=str(TSDF_ROOT / "data" / "extra_object"),
        help="Root directory containing one subdirectory per source category.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(TSDF_ROOT / "model" / "pointmlp" / "pointmlp_best_weights.pth"),
        help="PointMLP checkpoint path.",
    )
    parser.add_argument(
        "--labels",
        default=str(TSDF_ROOT / "model" / "pointmlp" / "labels.txt"),
        help="Class label file aligned with the checkpoint.",
    )
    parser.add_argument(
        "--output",
        default=str(TSDF_ROOT / "data" / "extra_object" / "pointmlp_scores.json"),
        help="Output JSON path.",
    )
    parser.add_argument("--num-points", type=int, default=None, help="Override num_points from checkpoint.")
    parser.add_argument("--num-votes", type=int, default=1, help="Average predictions over multiple random samplings.")
    parser.add_argument("--topk", type=int, default=5, help="How many top predictions to include per file.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for point sampling.")
    parser.add_argument("--use-all-points", action="store_true", help="Use all points instead of random sampling.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main():
    args = build_parser().parse_args()

    labels = load_labels(args.labels)
    checkpoint = load_raw_checkpoint(args.checkpoint, "cpu")
    model_type = checkpoint.get("model_type", "pointmlp") if isinstance(checkpoint, dict) else "pointmlp"
    ckpt_num_points = checkpoint.get("num_points", 1024) if isinstance(checkpoint, dict) else 1024
    num_points = args.num_points or ckpt_num_points

    model = PointMLPCls(k=len(labels), num_points=num_points, model_type=model_type).to(args.device)
    load_checkpoint(model, args.checkpoint, args.device)
    model.eval()

    category_to_files = collect_category_files(args.input_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    total_files = 0

    for source_category, files in category_to_files.items():
        category_results = []
        for file_path in files:
            raw_points = load_point_cloud_points(file_path)
            pred_idx, probs = predict_with_votes(
                model=model,
                points=raw_points,
                device=args.device,
                num_points=num_points,
                num_votes=args.num_votes,
                seed=args.seed,
                use_all_points=args.use_all_points,
            )
            category_results.append(
                build_result_record(
                    file_path=file_path,
                    source_category=source_category,
                    raw_points=raw_points,
                    labels=labels,
                    pred_idx=pred_idx,
                    probs=probs,
                    topk=args.topk,
                )
            )
            total_files += 1
            print(
                f"[{source_category}] {file_path.name} -> "
                f"{category_results[-1]['predicted_label']} "
                f"({category_results[-1]['confidence']:.4f})"
            )
        results[source_category] = category_results

    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "labels_path": str(Path(args.labels).resolve()),
        "input_root": str(Path(args.input_root).resolve()),
        "output_path": str(output_path.resolve()),
        "device": args.device,
        "model_type": model_type,
        "num_points": int(num_points),
        "num_votes": int(args.num_votes),
        "use_all_points": bool(args.use_all_points),
        "label_space": labels,
        "source_categories": list(results.keys()),
        "source_categories_in_label_space": [name for name in results if name in labels],
        "source_categories_outside_label_space": [name for name in results if name not in labels],
        "total_files": total_files,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"Saved JSON results to: {output_path.resolve()}")
    print(f"Processed {total_files} point clouds across {len(results)} source categories.")


if __name__ == "__main__":
    main()
