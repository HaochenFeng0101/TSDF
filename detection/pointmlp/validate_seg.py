import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
TSDF_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


'''

tsdf python detection/pointmlp/validate_seg.py \
  --pcd 3d_construction/outputs/Area_1_office_20.pcd \
  --checkpoint seg_model/pointmlp/pointmlp_seg_best.pth \
  --visualize

'''

from TSDF.detection.pointmlp.pointmlp_seg import PointMLPSemSeg, SEG_INPUT_CHANNELS
from TSDF.detection.pointnet2.validate_seg import (
    build_chunk_features,
    choose_chunk_indices,
    colorize_predictions,
    load_checkpoint,
    load_labels,
    load_scene,
    preprocess_scene,
    read_checkpoint_metadata,
    resolve_repo_path,
    write_colored_scene,
    write_masked_object,
)


DEFAULT_CHECKPOINT = Path("seg_model") / "pointmlp" / "pointmlp_seg_best.pth"
DEFAULT_LABELS_PATH = Path("seg_model") / "pointmlp" / "labels.txt"
DEFAULT_OUTPUT_DIR = Path("seg_model") / "pointmlp" / "inference_outputs"


def ensure_checkpoint_compatible(checkpoint):
    input_channels = int(checkpoint.get("input_channels", SEG_INPUT_CHANNELS))
    if input_channels != SEG_INPUT_CHANNELS:
        raise RuntimeError(
            f"Expected a {SEG_INPUT_CHANNELS}-channel segmentation checkpoint, got input_channels={input_channels}."
        )


def predict_scene(model, points, colors, device, num_points, min_votes, max_chunks):
    chunks = choose_chunk_indices(points, num_points, min_votes, max_chunks)
    if not chunks:
        raise ValueError("No chunks were generated for scene inference.")

    logits_sum = None
    logits_count = np.zeros(len(points), dtype=np.int32)
    room_norm_xyz = None

    model.eval()
    with torch.no_grad():
        for chunk_ids in chunks:
            if room_norm_xyz is None:
                mins = points.min(axis=0, keepdims=True)
                maxs = points.max(axis=0, keepdims=True)
                span = np.maximum(maxs - mins, 1e-6)
                room_norm_xyz = (points - mins) / span
            features = build_chunk_features(points, colors, room_norm_xyz, chunk_ids)
            tensor = torch.from_numpy(features.T).unsqueeze(0).to(device)
            scores = model(tensor).squeeze(0).transpose(0, 1).cpu().numpy()
            if logits_sum is None:
                logits_sum = np.zeros((len(points), scores.shape[1]), dtype=np.float32)
            logits_sum[chunk_ids] += scores
            logits_count[chunk_ids] += 1

    logits_count[logits_count == 0] = 1
    averaged_logits = logits_sum / logits_count[:, None]
    predictions = np.argmax(averaged_logits, axis=1).astype(np.int32)
    return predictions, averaged_logits, len(chunks)


def main():
    parser = argparse.ArgumentParser(
        description="Run PointMLP semantic segmentation on a scene point cloud."
    )
    parser.add_argument("--pcd", required=True, help="Scene point cloud path (.pcd, .ply, ...).")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="PointMLP segmentation checkpoint path. Defaults to seg_model/pointmlp/pointmlp_seg_best.pth.",
    )
    parser.add_argument(
        "--labels",
        default=str(DEFAULT_LABELS_PATH),
        help="Label file used to name predicted classes. Defaults to seg_model/pointmlp/labels.txt.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write segmentation outputs. Defaults to seg_model/pointmlp/inference_outputs.",
    )
    parser.add_argument("--target-class", default=None, help="Optional semantic class name to extract.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--model-type", choices=["pointmlp", "pointmlpelite"], default=None)
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
    model_type = args.model_type or ckpt.get("model_type", "pointmlp")

    model = PointMLPSemSeg(
        num_classes=num_classes,
        num_points=num_points,
        model_type=model_type,
        dropout=float(ckpt.get("dropout", 0.5)),
    ).to(args.device)
    load_checkpoint(model, checkpoint_path, args.device)
    labels = load_labels(labels_path, num_classes)

    scene, _, _ = load_scene(args.pcd)
    scene, points, colors = preprocess_scene(scene, args.voxel_size, args.max_points)
    predictions, _, num_chunks = predict_scene(
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
        "model_type": model_type,
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
        o3d.visualization.draw_geometries([colored], window_name="PointMLP Segmentation")


if __name__ == "__main__":
    main()
