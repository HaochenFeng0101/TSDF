import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.transforms.functional import to_tensor


TSDF_ROOT = Path(__file__).resolve().parents[1]


def update_recursive(dict1, dict2):
    for key, value in dict2.items():
        if key not in dict1:
            dict1[key] = {}
        if isinstance(value, dict):
            update_recursive(dict1[key], value)
        else:
            dict1[key] = value


def load_config(path):
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg_special = yaml.full_load(handle)
    inherit_from = cfg_special.get("inherit_from")
    if inherit_from is not None:
        inherit_path = Path(inherit_from)
        if not inherit_path.is_absolute():
            inherit_path = (config_path.parent / inherit_path).resolve()
        cfg = load_config(inherit_path)
    else:
        cfg = {}
    update_recursive(cfg, cfg_special)
    return cfg


def sanitize_name(name):
    return name.lower().replace(" ", "_").replace("/", "_")


def build_default_output_dir(config_path, target_classes):
    target_tag = "_".join(sanitize_name(name) for name in target_classes)
    return TSDF_ROOT / "mask_generation" / "outputs" / f"{Path(config_path).stem}_{target_tag}"


def build_model(device):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights)
    model.eval().to(device)
    categories = weights.meta["categories"]
    category_to_id = {name.lower(): idx for idx, name in enumerate(categories)}
    return model, categories, category_to_id


def sorted_frame_map(folder, suffixes):
    mapping = {}
    for suffix in suffixes:
        for path in folder.glob(f"*{suffix}"):
            mapping[path.stem] = path
    return dict(sorted(mapping.items(), key=lambda item: int(item[0])))


def load_scannet_rgb_entries(scene_path, frame_stride=1, max_frames=None):
    scene_path = Path(scene_path)
    color_dir = scene_path / "color"
    if not color_dir.exists():
        raise FileNotFoundError(
            f"Could not find ScanNet color folder: {color_dir}. "
            "Use the exported ScanNet scene path with color/depth/pose/intrinsic."
        )
    color_map = sorted_frame_map(color_dir, [".jpg", ".png"])
    frame_ids = list(color_map.keys())
    if frame_stride > 1:
        frame_ids = frame_ids[::frame_stride]
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
    return [(frame_id, color_map[frame_id]) for frame_id in frame_ids]


def ensure_mask_path(mask_root, frame_id):
    output_path = Path(mask_root) / "color" / f"{frame_id}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def ensure_track_mask_path(mask_root, track_id, frame_id):
    output_path = Path(mask_root) / f"track_{track_id:03d}" / "color" / f"{frame_id}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def ensure_preview_path(preview_root, frame_id):
    output_path = Path(preview_root) / "color" / f"{frame_id}.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def ensure_track_preview_path(preview_root, track_id, frame_id):
    output_path = (
        Path(preview_root) / f"track_{track_id:03d}" / "color" / f"{frame_id}.jpg"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def blend_preview(rgb, mask):
    overlay = rgb.copy()
    overlay[mask > 0] = (0.6 * overlay[mask > 0] + 0.4 * np.array([255, 0, 0])).astype(
        np.uint8
    )
    return overlay


def mask_center(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)


def mask_iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def assign_track_id(binary_mask, frame_index, tracks, args):
    center = mask_center(binary_mask)
    best_track_id = None
    best_iou = -1.0
    best_center_dist = None

    for track_id, track in tracks.items():
        if frame_index - track["last_frame_index"] > args.max_track_gap:
            continue

        prev_mask = track["last_mask"]
        iou = mask_iou(binary_mask, prev_mask)
        center_dist = np.inf
        if center is not None and track["last_center"] is not None:
            center_dist = float(np.linalg.norm(center - track["last_center"]))

        if iou < args.track_iou_threshold and center_dist > args.track_center_threshold:
            continue

        if iou > best_iou or (
            np.isclose(iou, best_iou)
            and best_center_dist is not None
            and center_dist < best_center_dist
        ):
            best_track_id = track_id
            best_iou = iou
            best_center_dist = center_dist

    if best_track_id is None:
        best_track_id = len(tracks)

    tracks[best_track_id] = {
        "last_frame_index": frame_index,
        "last_mask": binary_mask.copy(),
        "last_center": center,
    }
    return best_track_id


def generate_masks(args):
    config = load_config(args.config)
    scene_path = Path(args.dataset or config["Dataset"]["dataset_path"])
    rgb_entries = load_scannet_rgb_entries(
        scene_path=scene_path,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )

    output_dir = Path(args.output_dir) if args.output_dir else build_default_output_dir(
        args.config, args.target_class
    )
    mask_root = output_dir / "masks"
    preview_root = output_dir / "preview"
    mask_root.mkdir(parents=True, exist_ok=True)
    if args.save_preview:
        preview_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, categories, category_to_id = build_model(device)

    target_ids = []
    for class_name in args.target_class:
        key = class_name.lower()
        if key not in category_to_id:
            raise ValueError(
                f"Unknown COCO class '{class_name}'. Available examples include: "
                f"{', '.join(categories[1:15])}"
            )
        target_ids.append(category_to_id[key])

    metadata = {
        "config": str(args.config),
        "dataset": str(scene_path),
        "target_class": args.target_class,
        "score_threshold": args.score_threshold,
        "mask_threshold": args.mask_threshold,
        "merge_mode": args.merge_mode,
        "device": str(device),
        "frames_total": int(len(rgb_entries)),
        "frames_with_detection": 0,
        "frames_without_detection": 0,
        "frames_saved": 0,
        "separate_instances": bool(args.separate_instances),
    }

    mask_txt_path = output_dir / "mask.txt"
    detections_jsonl = output_dir / "detections.jsonl"
    tracks = {}
    track_lines = {}
    track_stats = {}

    with open(mask_txt_path, "w", encoding="utf-8") as mask_txt, open(
        detections_jsonl, "w", encoding="utf-8"
    ) as detections_file:
        for idx, (frame_id, rgb_path) in enumerate(rgb_entries, start=1):
            image = Image.open(rgb_path).convert("RGB")
            image_np = np.array(image)
            image_tensor = to_tensor(image).to(device)

            with torch.inference_mode():
                prediction = model([image_tensor])[0]

            labels = prediction["labels"].detach().cpu().numpy()
            scores = prediction["scores"].detach().cpu().numpy()
            masks = prediction["masks"].detach().cpu().numpy()

            selected = []
            for det_idx, (label, score) in enumerate(zip(labels, scores)):
                if label in target_ids and score >= args.score_threshold:
                    selected.append(det_idx)

            binary_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
            frame_summary = {
                "frame_id": frame_id,
                "rgb_path": str(rgb_path),
                "selected_count": len(selected),
                "detections": [],
            }

            if selected:
                if args.separate_instances:
                    metadata["frames_with_detection"] += 1
                elif args.merge_mode == "top1":
                    selected = [selected[int(np.argmax(scores[selected]))]]
                    selected_masks = masks[selected, 0]
                    binary_mask = (
                        np.max(selected_masks, axis=0) >= args.mask_threshold
                    ).astype(np.uint8) * 255
                    metadata["frames_with_detection"] += 1
                else:
                    selected_masks = masks[selected, 0]
                    binary_mask = (
                        np.max(selected_masks, axis=0) >= args.mask_threshold
                    ).astype(np.uint8) * 255
                    metadata["frames_with_detection"] += 1
            else:
                metadata["frames_without_detection"] += 1

            for det_idx in selected:
                detection_summary = {
                    "label": categories[int(labels[det_idx])],
                    "score": float(scores[det_idx]),
                }
                if args.separate_instances:
                    instance_mask = (masks[det_idx, 0] >= args.mask_threshold).astype(bool)
                    if instance_mask.any():
                        track_id = assign_track_id(instance_mask, idx, tracks, args)
                        detection_summary["track_id"] = int(track_id)

                        output_mask_path = ensure_track_mask_path(mask_root, track_id, frame_id)
                        cv2.imwrite(str(output_mask_path), instance_mask.astype(np.uint8) * 255)
                        relative_mask_path = output_mask_path.relative_to(output_dir)
                        track_lines.setdefault(track_id, []).append(
                            f"{frame_id} {relative_mask_path.as_posix()}\n"
                        )
                        stats = track_stats.setdefault(track_id, {"frames_saved": 0})
                        stats["frames_saved"] += 1
                        metadata["frames_saved"] += 1

                        if args.save_preview:
                            preview_path = ensure_track_preview_path(
                                preview_root, track_id, frame_id
                            )
                            cv2.imwrite(
                                str(preview_path),
                                cv2.cvtColor(
                                    blend_preview(
                                        image_np, instance_mask.astype(np.uint8) * 255
                                    ),
                                    cv2.COLOR_RGB2BGR,
                                ),
                            )
                frame_summary["detections"].append(detection_summary)

            if selected and not args.separate_instances:
                output_mask_path = ensure_mask_path(mask_root, frame_id)
                cv2.imwrite(str(output_mask_path), binary_mask)
                relative_mask_path = output_mask_path.relative_to(output_dir)
                mask_txt.write(f"{frame_id} {relative_mask_path.as_posix()}\n")
                metadata["frames_saved"] += 1

                if args.save_preview:
                    preview_path = ensure_preview_path(preview_root, frame_id)
                    cv2.imwrite(
                        str(preview_path),
                        cv2.cvtColor(blend_preview(image_np, binary_mask), cv2.COLOR_RGB2BGR),
                    )

            detections_file.write(json.dumps(frame_summary) + "\n")

            if idx % args.log_every == 0 or idx == len(rgb_entries):
                print(
                    f"Processed {idx}/{len(rgb_entries)} frames | "
                    f"with_detection={metadata['frames_with_detection']} "
                    f"without_detection={metadata['frames_without_detection']}"
                )

    if args.separate_instances:
        track_summaries = []
        for track_id, lines in sorted(track_lines.items()):
            track_mask_txt_path = output_dir / f"mask_track_{track_id:03d}.txt"
            with open(track_mask_txt_path, "w", encoding="utf-8") as handle:
                handle.writelines(lines)
            track_summaries.append(
                {
                    "track_id": track_id,
                    "frames_saved": track_stats[track_id]["frames_saved"],
                    "mask_list": str(track_mask_txt_path),
                }
            )
        metadata["tracks"] = track_summaries
    else:
        metadata["tracks"] = []

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved masks to {mask_root}")
    print(f"Saved mask index to {mask_txt_path}")
    print(f"Saved detection log to {detections_jsonl}")
    print(f"Saved metadata to {metadata_path}")
    print(f"Saved frames with masks: {metadata['frames_saved']}")


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Generate ScanNet scene masks with torchvision Mask R-CNN."
    )
    parser.add_argument(
        "--config",
        default=str(TSDF_ROOT / "configs" / "rgbd" / "scannet" / "scene0000_00.yaml"),
        help="Path to a ScanNet config file.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override exported ScanNet scene path from the config file.",
    )
    parser.add_argument(
        "--target-class",
        action="append",
        required=True,
        help="COCO target class to segment. Repeat the flag for multiple classes.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save masks. Defaults to mask_generation/outputs/<config>_<target>.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, e.g. cpu or cuda. Defaults to cuda if available.",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument(
        "--merge-mode",
        choices=("union", "top1"),
        default="top1",
        help="Use the top scoring instance or merge all kept instances into one mask.",
    )
    parser.add_argument(
        "--separate-instances",
        action="store_true",
        help="Save same-class detections as separate instance tracks instead of merging them.",
    )
    parser.add_argument("--max-track-gap", type=int, default=10)
    parser.add_argument("--track-iou-threshold", type=float, default=0.05)
    parser.add_argument("--track-center-threshold", type=float, default=120.0)
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    generate_masks(parser.parse_args())
