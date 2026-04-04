import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

'''

 python3 mask_generation/generate_tum_masks_yolo.py \
  --config configs/rgbd/tum/fr3_office.yaml \
  --model yolov8x-seg.pt \
  --target-class book
'''
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


def parse_list(filepath):
    return np.loadtxt(filepath, delimiter=" ", dtype=str)


def sanitize_name(name):
    return name.lower().replace(" ", "_").replace("/", "_")


def build_default_output_dir(config_path, target_classes, model_name):
    target_tag = "_".join(sanitize_name(name) for name in target_classes)
    model_tag = sanitize_name(model_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return TSDF_ROOT / "mask_generation" / "outputs" / f"{Path(config_path).stem}_{target_tag}_{model_tag}_{timestamp}"


def load_rgb_entries(dataset_path, frame_stride=1, max_frames=None):
    image_list = os.path.join(dataset_path, "rgb.txt")
    if not os.path.isfile(image_list):
        raise FileNotFoundError(f"Could not find rgb.txt in {dataset_path}")
    image_data = parse_list(image_list)
    if image_data.ndim == 1:
        image_data = image_data.reshape(1, -1)
    if frame_stride > 1:
        image_data = image_data[::frame_stride]
    if max_frames is not None:
        image_data = image_data[:max_frames]
    return image_data


def ensure_rgb_mask_path(mask_root, rgb_relpath):
    relative = Path(rgb_relpath)
    output_path = Path(mask_root) / relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def ensure_track_rgb_mask_path(mask_root, track_id, rgb_relpath):
    relative = Path(rgb_relpath)
    output_path = Path(mask_root) / f"track_{track_id:03d}" / relative
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


def build_model(model_path):
    if YOLO is None:
        raise RuntimeError(
            "ultralytics is not installed. Install it first to run YOLO mask generation."
        )
    model = YOLO(model_path)
    names = model.names
    if isinstance(names, list):
        categories = names
    else:
        categories = [names[idx] for idx in sorted(names)]
    category_to_id = {str(name).lower(): idx for idx, name in enumerate(categories)}
    return model, categories, category_to_id


def select_device(device):
    if device:
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def normalize_target_name(class_name):
    return class_name.strip().lower()


def run_inference(model, image_path, args):
    results = model.predict(
        source=str(image_path),
        device=select_device(args.device),
        conf=args.score_threshold,
        iou=args.nms_iou,
        retina_masks=True,
        verbose=False,
    )
    if not results:
        return None
    return results[0]


def result_to_instances(result, categories):
    if result.boxes is None or result.boxes.cls is None or len(result.boxes) == 0:
        return []

    if result.masks is None or result.masks.data is None:
        raise RuntimeError(
            "The selected YOLO model did not return segmentation masks. "
            "Use a segmentation checkpoint, for example a '*-seg' model."
        )

    boxes = result.boxes
    masks = result.masks.data.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(np.int64)
    scores = boxes.conf.detach().cpu().numpy().astype(np.float32)
    instances = []
    for det_idx, cls_idx in enumerate(classes):
        label = str(categories[int(cls_idx)])
        mask = masks[det_idx] > 0.5
        instances.append(
            {
                "label": label,
                "score": float(scores[det_idx]),
                "mask": mask,
            }
        )
    return instances


def cleanup_rejected_track_outputs(output_dir, preview_root, track_id):
    track_mask_dir = Path(output_dir) / "masks" / f"track_{track_id:03d}"
    if track_mask_dir.exists():
        shutil.rmtree(track_mask_dir, ignore_errors=True)
    if preview_root is not None:
        track_preview_dir = Path(preview_root) / f"track_{track_id:03d}"
        if track_preview_dir.exists():
            shutil.rmtree(track_preview_dir, ignore_errors=True)


def generate_masks(args):
    config = load_config(args.config)
    dataset_path = args.dataset or config["Dataset"]["dataset_path"]
    image_entries = load_rgb_entries(
        dataset_path,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )

    model_name = Path(args.model).stem
    output_dir = Path(args.output_dir) if args.output_dir else build_default_output_dir(
        args.config, args.target_class, model_name
    )
    mask_root = output_dir / "masks"
    preview_root = output_dir / "preview"
    mask_root.mkdir(parents=True, exist_ok=True)
    if args.save_preview:
        preview_root.mkdir(parents=True, exist_ok=True)

    model, categories, category_to_id = build_model(args.model)
    target_ids = []
    for class_name in args.target_class:
        key = normalize_target_name(class_name)
        if key not in category_to_id:
            raise ValueError(
                f"Unknown YOLO class '{class_name}'. Available examples include: "
                f"{', '.join(categories[:15])}"
            )
        target_ids.append(category_to_id[key])

    metadata = {
        "config": str(args.config),
        "dataset": str(dataset_path),
        "target_class": args.target_class,
        "model": str(args.model),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "score_threshold": args.score_threshold,
        "merge_mode": args.merge_mode,
        "device": select_device(args.device),
        "frames_total": int(len(image_entries)),
        "frames_with_detection": 0,
        "frames_without_detection": 0,
        "frames_saved": 0,
        "separate_instances": bool(args.separate_instances),
        "min_instance_mask_pixels": int(args.min_instance_mask_pixels),
        "min_track_frames": int(args.min_track_frames),
    }

    mask_txt_path = output_dir / "mask.txt"
    detections_jsonl = output_dir / "detections.jsonl"
    tracks = {}
    track_lines = {}
    track_stats = {}

    with open(mask_txt_path, "w", encoding="utf-8") as mask_txt, open(
        detections_jsonl, "w", encoding="utf-8"
    ) as detections_file:
        for idx, entry in enumerate(image_entries, start=1):
            timestamp = str(entry[0])
            rgb_relpath = entry[1]
            rgb_path = Path(dataset_path) / rgb_relpath
            if not rgb_path.is_file():
                raise FileNotFoundError(f"Could not find rgb frame: {rgb_path}")

            image = Image.open(rgb_path).convert("RGB")
            image_np = np.array(image)
            result = run_inference(model, rgb_path, args)
            instances = [] if result is None else result_to_instances(result, categories)

            selected = [
                det_idx
                for det_idx, instance in enumerate(instances)
                if normalize_target_name(instance["label"]) in {normalize_target_name(name) for name in args.target_class}
                and instance["score"] >= args.score_threshold
                and int(instance["mask"].sum()) >= args.min_instance_mask_pixels
            ]

            binary_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
            frame_summary = {
                "timestamp": timestamp,
                "rgb_path": rgb_relpath,
                "selected_count": len(selected),
                "detections": [],
            }

            if selected:
                if args.separate_instances:
                    metadata["frames_with_detection"] += 1
                elif args.merge_mode == "top1":
                    top_idx = max(selected, key=lambda det_idx: instances[det_idx]["score"])
                    selected = [top_idx]
                    binary_mask = instances[top_idx]["mask"].astype(np.uint8) * 255
                    metadata["frames_with_detection"] += 1
                else:
                    selected_masks = [instances[det_idx]["mask"] for det_idx in selected]
                    binary_mask = np.logical_or.reduce(selected_masks).astype(np.uint8) * 255
                    metadata["frames_with_detection"] += 1
            else:
                metadata["frames_without_detection"] += 1

            for det_idx in selected:
                instance = instances[det_idx]
                detection_summary = {
                    "label": instance["label"],
                    "score": instance["score"],
                }
                if args.separate_instances:
                    instance_mask = instance["mask"].astype(bool)
                    if instance_mask.any():
                        track_id = assign_track_id(instance_mask, idx, tracks, args)
                        detection_summary["track_id"] = int(track_id)

                        output_mask_path = ensure_track_rgb_mask_path(
                            mask_root, track_id, rgb_relpath
                        )
                        cv2.imwrite(str(output_mask_path), instance_mask.astype(np.uint8) * 255)
                        relative_mask_path = output_mask_path.relative_to(output_dir)
                        track_lines.setdefault(track_id, []).append(
                            f"{timestamp} {relative_mask_path.as_posix()}\n"
                        )
                        stats = track_stats.setdefault(track_id, {"frames_saved": 0})
                        stats["frames_saved"] += 1
                        metadata["frames_saved"] += 1

                        if args.save_preview:
                            preview_path = ensure_track_rgb_mask_path(
                                preview_root, track_id, rgb_relpath
                            ).with_suffix(".jpg")
                            cv2.imwrite(
                                str(preview_path),
                                cv2.cvtColor(
                                    blend_preview(image_np, instance_mask.astype(np.uint8) * 255),
                                    cv2.COLOR_RGB2BGR,
                                ),
                            )
                frame_summary["detections"].append(detection_summary)

            if selected and not args.separate_instances:
                output_mask_path = ensure_rgb_mask_path(mask_root, rgb_relpath)
                cv2.imwrite(str(output_mask_path), binary_mask)
                relative_mask_path = output_mask_path.relative_to(output_dir)
                mask_txt.write(f"{timestamp} {relative_mask_path.as_posix()}\n")
                metadata["frames_saved"] += 1

                if args.save_preview:
                    preview_path = ensure_rgb_mask_path(preview_root, rgb_relpath).with_suffix(
                        ".jpg"
                    )
                    cv2.imwrite(
                        str(preview_path),
                        cv2.cvtColor(blend_preview(image_np, binary_mask), cv2.COLOR_RGB2BGR),
                    )

            detections_file.write(json.dumps(frame_summary) + "\n")

            if idx % args.log_every == 0 or idx == len(image_entries):
                print(
                    f"Processed {idx}/{len(image_entries)} frames | "
                    f"with_detection={metadata['frames_with_detection']} "
                    f"without_detection={metadata['frames_without_detection']}"
                )

    if args.separate_instances:
        track_summaries = []
        rejected_tracks = []
        for track_id, lines in sorted(track_lines.items()):
            frames_saved = track_stats[track_id]["frames_saved"]
            if frames_saved < args.min_track_frames:
                cleanup_rejected_track_outputs(
                    output_dir=output_dir,
                    preview_root=preview_root if args.save_preview else None,
                    track_id=track_id,
                )
                rejected_tracks.append(
                    {
                        "track_id": track_id,
                        "frames_saved": frames_saved,
                        "reason": f"frames_saved < {args.min_track_frames}",
                    }
                )
                continue
            track_mask_txt_path = output_dir / f"mask_track_{track_id:03d}.txt"
            with open(track_mask_txt_path, "w", encoding="utf-8") as handle:
                handle.writelines(lines)
            track_summaries.append(
                {
                    "track_id": track_id,
                    "frames_saved": frames_saved,
                    "mask_list": str(track_mask_txt_path),
                }
            )
        metadata["tracks"] = track_summaries
        metadata["rejected_tracks"] = rejected_tracks
    else:
        metadata["tracks"] = []
        metadata["rejected_tracks"] = []

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
        description="Generate TUM RGB-D object masks with a YOLO segmentation model."
    )
    parser.add_argument(
        "--config",
        default=str(TSDF_ROOT / "configs" / "rgbd" / "tum" / "fr3_office.yaml"),
        help="Path to a TUM RGB-D config file.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override dataset path from the config file.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a YOLO segmentation checkpoint, for example yolov9c-seg.pt or a custom weight file.",
    )
    parser.add_argument(
        "--target-class",
        action="append",
        required=True,
        help="Target class to segment. Repeat the flag for multiple classes.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save masks. Defaults to a timestamped folder under mask_generation/outputs/<config>_<target>_<model>_<timestamp>.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, e.g. cpu or cuda:0. Defaults to cuda:0 if available.",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument(
        "--min-instance-mask-pixels",
        type=int,
        default=2500,
        help="Ignore small masks below this pixel count before tracking or saving.",
    )
    parser.add_argument(
        "--merge-mode",
        choices=("union", "top1"),
        default="top1",
        help="Use the top scoring instance or merge all kept instances into one mask.",
    )
    parser.add_argument(
        "--separate-instances",
        dest="separate_instances",
        action="store_true",
        help="Save same-class detections as separate instance tracks. Enabled by default.",
    )
    parser.add_argument(
        "--merge-instances",
        dest="separate_instances",
        action="store_false",
        help="Merge same-class detections instead of saving them as separate instance tracks.",
    )
    parser.add_argument("--max-track-gap", type=int, default=10)
    parser.add_argument("--track-iou-threshold", type=float, default=0.05)
    parser.add_argument("--track-center-threshold", type=float, default=120.0)
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=60,
        help="Ignore tracked objects shorter than this many saved mask frames.",
    )
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.set_defaults(separate_instances=True)
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    generate_masks(parser.parse_args())
