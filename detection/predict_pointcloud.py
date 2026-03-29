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

from TSDF.detection.pointmlp.model import PointMLPCls
from TSDF.detection.modelnet40c.models import (
    MODEL_SPECS,
    build_model as build_official_model,
    normalize_model_name,
)
from TSDF.detection.pointnet_model import PointNetCls
from TSDF.detection.train_pointnet_cls import load_point_cloud_file, normalize_points
from TSDF.dataset.modelnet40_data import load_off_points


SUPPORTED_MODELS = ("auto",) + tuple(sorted(MODEL_SPECS))
GRAPH_HEAVY_MODELS = {"curvenet", "dgcnn", "gdanet", "pct", "pointnet2", "rscnn"}


def load_labels(labels_path):
    with open(labels_path, "r", encoding="utf-8") as handle:
        labels = [line.strip() for line in handle if line.strip()]
    if not labels:
        raise ValueError(f"No labels found in {labels_path}")
    return labels


def ensure_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


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


def infer_model_type(checkpoint, cleaned_state_dict):
    if isinstance(checkpoint, dict):
        for field in ("model_name", "model_type"):
            model_type = checkpoint.get(field)
            if isinstance(model_type, str) and model_type.strip():
                return normalize_model_name(model_type)

    keys = list(cleaned_state_dict)
    if any(key.startswith("feat.") for key in keys):
        return "pointnet"
    if any(key.startswith("model.embedding.") for key in keys):
        return "pointmlp"
    raise RuntimeError(
        "Could not infer model type from checkpoint. "
        f"Use --model to specify one of: {', '.join(SUPPORTED_MODELS[1:])}."
    )


def resolve_model_type(requested_model, checkpoint, cleaned_state_dict):
    inferred_model = infer_model_type(checkpoint, cleaned_state_dict)
    requested_model = normalize_model_name(requested_model)
    if requested_model == "auto":
        return inferred_model
    if requested_model == inferred_model:
        return requested_model
    if requested_model == "pointmlp" and inferred_model == "pointmlpelite":
        return inferred_model
    raise RuntimeError(
        f"Requested --model {requested_model} does not match checkpoint model type {inferred_model}. "
        f"Use --model {inferred_model} or omit --model."
    )


def is_wrapped_checkpoint(cleaned_state_dict):
    return any(key.startswith("model.") for key in cleaned_state_dict)


def build_model(model_type, num_classes, num_points, checkpoint, cleaned_state_dict):
    normalized = normalize_model_name(model_type)
    if normalized == "pointnet":
        if is_wrapped_checkpoint(cleaned_state_dict):
            return (
                build_official_model(normalized, num_classes=num_classes, num_points=num_points),
                "PointNet",
            )
        dropout = checkpoint.get("dropout", 0.3) if isinstance(checkpoint, dict) else 0.3
        return PointNetCls(k=num_classes, dropout=dropout), "PointNet"
    if normalized in {"pointmlp", "pointmlpelite"}:
        return (
            PointMLPCls(k=num_classes, num_points=num_points, model_type=normalized),
            "PointMLP-Elite" if normalized == "pointmlpelite" else "PointMLP",
        )
    if normalized in MODEL_SPECS:
        return (
            build_official_model(normalized, num_classes=num_classes, num_points=num_points),
            MODEL_SPECS[normalized]["label"],
        )
    raise RuntimeError(f"Unsupported model type: {model_type}")


def load_checkpoint(model, cleaned_state_dict):
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing checkpoint keys: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")


def run_model(model, tensor, model_type=None):
    input_tensor = tensor
    # CurveNet's upstream implementation can break on batch_size=1 because of a squeeze.
    if model_type == "curvenet" and tensor.shape[0] == 1:
        input_tensor = tensor.repeat(2, 1, 1)
    outputs = model(input_tensor)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    if model_type == "curvenet" and tensor.shape[0] == 1:
        outputs = outputs[:1]
    return outputs


def predict_with_votes(model, points, device, num_points, num_votes, use_all_points, seed, model_type):
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
            logits = run_model(model, tensor, model_type=model_type)
        logits_votes.append(logits[0].detach().cpu())
    mean_logits = torch.stack(logits_votes, dim=0).mean(dim=0)
    probs = torch.softmax(mean_logits, dim=0)
    return probs


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


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a trained point-cloud classifier checkpoint on one point cloud or ModelNet40 .off file."
    )
    parser.add_argument("--input", required=True, help="Input point file: .off/.pcd/.ply/.npy/.npz/.txt/.xyz/.pts")
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        default=[str(TSDF_ROOT / "model" / "pointnet" / "pointnet_best.pth")],
        help="One or more checkpoint paths for PointNet / PointMLP / CurveNet / DGCNN / other supported models.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=[str(TSDF_ROOT / "model" / "pointnet" / "labels.txt")],
        help="One or more label files, aligned with --checkpoint.",
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        nargs="+",
        default=["auto"],
        help="One or more model names aligned with --checkpoint. Use auto to infer from each checkpoint.",
    )
    parser.add_argument("--num-points", type=int, default=None, help="Override num_points from checkpoint.")
    parser.add_argument("--num-votes", type=int, default=1, help="Average predictions over multiple random point samplings.")
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
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    checkpoint_paths = ensure_list(args.checkpoint)
    label_paths = ensure_list(args.labels)
    requested_models = ensure_list(args.model)

    num_runs = len(checkpoint_paths)
    if len(label_paths) == 1 and num_runs > 1:
        label_paths = label_paths * num_runs
    if len(requested_models) == 1 and num_runs > 1:
        requested_models = requested_models * num_runs
    if len(label_paths) != num_runs:
        raise ValueError("--labels must provide either 1 path or the same number of paths as --checkpoint.")
    if len(requested_models) != num_runs:
        raise ValueError("--model must provide either 1 value or the same number of values as --checkpoint.")

    prepared_results = []
    for index, checkpoint_path in enumerate(checkpoint_paths):
        labels = load_labels(label_paths[index])
        checkpoint = load_raw_checkpoint(checkpoint_path, args.device)
        cleaned_state_dict = clean_state_dict(checkpoint)
        num_points = args.num_points or checkpoint.get("num_points", 1024)
        model_type = resolve_model_type(requested_models[index], checkpoint, cleaned_state_dict)
        if args.use_all_points and model_type in GRAPH_HEAVY_MODELS:
            raise RuntimeError(
                f"--use-all-points is not supported for {model_type} because memory usage grows "
                "too quickly with large point clouds. Remove --use-all-points and use the "
                "checkpoint's default num_points or pass --num-points explicitly."
            )
        model, model_label = build_model(model_type, len(labels), num_points, checkpoint, cleaned_state_dict)
        model = model.to(args.device)
        load_checkpoint(model, cleaned_state_dict)
        model.eval()
        prepared_results.append(
            {
                "checkpoint_path": checkpoint_path,
                "label_path": label_paths[index],
                "labels": labels,
                "checkpoint": checkpoint,
                "num_points": num_points,
                "model_type": model_type,
                "model_label": model_label,
                "model": model,
            }
        )

    for index, item in enumerate(prepared_results):
        raw_points = load_input_points(
            args.input,
            num_points=item["num_points"],
            seed=args.seed,
            modelnet40_sample_method=args.modelnet40_sample_method,
        )
        probs = predict_with_votes(
            model=item["model"],
            points=raw_points,
            device=args.device,
            num_points=item["num_points"],
            num_votes=args.num_votes,
            use_all_points=args.use_all_points,
            seed=args.seed,
            model_type=item["model_type"],
        )
        pred_idx = int(torch.argmax(probs).item())
        topk = min(5, len(item["labels"]))
        top_probs, top_indices = torch.topk(probs, k=topk)

        if index > 0:
            print()
        print(f"input: {args.input}")
        print(f"checkpoint: {item['checkpoint_path']}")
        print(f"model: {item['model_type']}")
        print(f"raw_num_points: {len(raw_points)}")
        print(f"evaluation_mode: {'all_points' if args.use_all_points else f'{args.num_votes}_vote_sampling'}")
        print(f"predicted: {item['labels'][pred_idx]}")
        print(f"confidence: {float(probs[pred_idx]):.4f}")
        print("top_predictions:")
        for rank in range(topk):
            cls_idx = int(top_indices[rank].item())
            score = float(top_probs[rank].item())
            print(f"  {rank + 1}. {item['labels'][cls_idx]} ({score:.4f})")

        if args.visualize:
            points_to_show = raw_points if args.visualize_raw_points else prepare_points(
                raw_points,
                num_points=item["num_points"],
                seed=args.seed,
                use_all_points=args.use_all_points,
            )
            title = f"{item['model_label']} | pred={item['labels'][pred_idx]}"
            visualize_point_cloud(points_to_show, title)


if __name__ == "__main__":
    main()
