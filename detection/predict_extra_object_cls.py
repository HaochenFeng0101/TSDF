import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TSDF_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MODEL_CHOICES = ("pointnet", "pointnext", "pointnet2")
SUPPORTED_SUFFIXES = {".pcd", ".ply", ".xyz", ".pts", ".txt", ".npy", ".npz", ".off"}

MODEL_DEFAULTS = {
    "pointnet": {
        "checkpoint": TSDF_ROOT / "model" / "pointnet" / "pointnet_best.pth",
        "labels": TSDF_ROOT / "model" / "pointnet" / "labels.txt",
        "output": TSDF_ROOT / "data" / "extra_object" / "pointnet_scores.json",
    },
    "pointnext": {
        "checkpoint": TSDF_ROOT / "model" / "pointnext" / "pointnext_best.pth",
        "labels": TSDF_ROOT / "model" / "pointnext" / "labels.txt",
        "output": TSDF_ROOT / "data" / "extra_object" / "pointnext_scores.json",
    },
    "pointnet2": {
        "checkpoint": TSDF_ROOT / "model" / "pointnet2" / "pointnet2_best.pth",
        "labels": TSDF_ROOT / "model" / "pointnet2" / "labels.txt",
        "output": TSDF_ROOT / "data" / "extra_object" / "pointnet2_scores.json",
    },
}


def load_raw_checkpoint(checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def infer_num_classes_from_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        return None
    labels = checkpoint.get("labels")
    if isinstance(labels, list) and labels:
        return len(labels)
    return None


def collect_category_files(input_root):
    input_root = Path(input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    category_to_files = {}
    for category_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        files = sorted(
            path
            for path in category_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        category_to_files[category_dir.name] = files
    return category_to_files


def build_result_record(file_path, source_category, raw_points, labels, pred_idx, probs, topk):
    topk = min(topk, len(labels))
    top_probs, top_indices = torch.topk(probs, k=topk)

    score_map = {label: float(probs[idx].item()) for idx, label in enumerate(labels)}
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


def get_model_backend(model_name):
    model_name = model_name.lower()
    if model_name == "pointnet":
        from TSDF.detection.pointnet.pointnet_cls import PointNetCls
        from TSDF.detection.pointnet.validate import (
            load_checkpoint,
            load_labels,
            load_point_cloud_points,
            predict_with_votes,
        )

        def build_model(num_classes, checkpoint):
            _ = checkpoint
            return PointNetCls(k=num_classes)

        return build_model, load_checkpoint, load_labels, load_point_cloud_points, predict_with_votes

    if model_name == "pointnext":
        from TSDF.detection.pointnext.pointnext_cls import PointNeXtSmallCls
        from TSDF.detection.pointnext.validate import (
            load_checkpoint,
            load_labels,
            load_point_cloud_points,
            predict_with_votes,
        )

        def build_model(num_classes, checkpoint):
            dropout = checkpoint.get("dropout", 0.4) if isinstance(checkpoint, dict) else 0.4
            return PointNeXtSmallCls(num_classes=num_classes, input_channels=3, dropout=dropout)

        return build_model, load_checkpoint, load_labels, load_point_cloud_points, predict_with_votes

    if model_name == "pointnet2":
        from TSDF.detection.pointnet2.pointnet2 import PointNet2ClsSSG
        from TSDF.detection.pointnet2.validate import (
            load_checkpoint,
            load_labels,
            load_point_cloud_points,
            predict_with_votes,
        )

        def build_model(num_classes, checkpoint):
            _ = checkpoint
            return PointNet2ClsSSG(num_classes=num_classes)

        return build_model, load_checkpoint, load_labels, load_point_cloud_points, predict_with_votes

    raise ValueError(f"Unsupported model: {model_name}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Batch-run PointNet/PointNeXt/PointNet2 on all point clouds under data/extra_object and save JSON."
    )
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument(
        "--input-root",
        default=str(TSDF_ROOT / "data" / "extra_object"),
        help="Root directory containing one subdirectory per source category.",
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Uses model-specific default if omitted.")
    parser.add_argument("--labels", default=None, help="Label file path. Uses model-specific default if omitted.")
    parser.add_argument("--output", default=None, help="Output JSON path. Uses model-specific default if omitted.")
    parser.add_argument("--num-points", type=int, default=None, help="Override num_points from checkpoint.")
    parser.add_argument("--num-votes", type=int, default=1, help="Average predictions over multiple random samplings.")
    parser.add_argument("--topk", type=int, default=5, help="How many top predictions to include per file.")
    parser.add_argument("--use-all-points", action="store_true", help="Use all points instead of random sampling.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main():
    args = build_parser().parse_args()
    defaults = MODEL_DEFAULTS[args.model]

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else defaults["checkpoint"]
    labels_path = Path(args.labels) if args.labels else defaults["labels"]
    output_path = Path(args.output) if args.output else defaults["output"]

    build_model, load_checkpoint, load_labels, load_point_cloud_points, predict_with_votes = get_model_backend(
        args.model
    )

    labels = load_labels(str(labels_path))
    checkpoint = load_raw_checkpoint(checkpoint_path, "cpu")
    ckpt_num_points = checkpoint.get("num_points", 1024) if isinstance(checkpoint, dict) else 1024
    num_points = args.num_points or ckpt_num_points

    ckpt_num_classes = infer_num_classes_from_checkpoint(checkpoint)
    if ckpt_num_classes is not None and ckpt_num_classes != len(labels):
        raise RuntimeError(
            f"Label count ({len(labels)}) does not match checkpoint classes ({ckpt_num_classes})."
        )

    model = build_model(len(labels), checkpoint).to(args.device)
    load_checkpoint(model, str(checkpoint_path), args.device)
    model.eval()

    category_to_files = collect_category_files(args.input_root)
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
                use_all_points=args.use_all_points,
            )

            record = build_result_record(
                file_path=file_path,
                source_category=source_category,
                raw_points=raw_points,
                labels=labels,
                pred_idx=pred_idx,
                probs=probs,
                topk=args.topk,
            )
            category_results.append(record)
            total_files += 1
            print(f"[{source_category}] {file_path.name} -> {record['predicted_label']} ({record['confidence']:.4f})")

        results[source_category] = category_results

    payload = {
        "model": args.model,
        "checkpoint": str(checkpoint_path.resolve()),
        "labels_path": str(labels_path.resolve()),
        "input_root": str(Path(args.input_root).resolve()),
        "output_path": str(output_path.resolve()),
        "device": args.device,
        "num_points": int(num_points),
        "num_votes": int(args.num_votes),
        "use_all_points": bool(args.use_all_points),
        "topk": int(args.topk),
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
