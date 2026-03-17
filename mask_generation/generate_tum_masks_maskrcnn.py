import argparse
import json
import os
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
    with open(path, "r", encoding="utf-8") as handle:
        cfg_special = yaml.full_load(handle)
    inherit_from = cfg_special.get("inherit_from")
    cfg = load_config(inherit_from) if inherit_from is not None else {}
    update_recursive(cfg, cfg_special)
    return cfg


def parse_list(filepath):
    return np.loadtxt(filepath, delimiter=" ", dtype=np.str_)


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


def blend_preview(rgb, mask):
    overlay = rgb.copy()
    overlay[mask > 0] = (0.6 * overlay[mask > 0] + 0.4 * np.array([255, 0, 0])).astype(
        np.uint8
    )
    return overlay


def generate_masks(args):
    config = load_config(args.config)
    dataset_path = args.dataset or config["Dataset"]["dataset_path"]
    image_entries = load_rgb_entries(
        dataset_path,
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
        "dataset": str(dataset_path),
        "target_class": args.target_class,
        "score_threshold": args.score_threshold,
        "mask_threshold": args.mask_threshold,
        "merge_mode": args.merge_mode,
        "device": str(device),
        "frames_total": int(len(image_entries)),
        "frames_with_detection": 0,
        "frames_without_detection": 0,
        "frames_saved": 0,
    }

    mask_txt_path = output_dir / "mask.txt"
    detections_jsonl = output_dir / "detections.jsonl"

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
                "timestamp": timestamp,
                "rgb_path": rgb_relpath,
                "selected_count": len(selected),
                "detections": [],
            }

            if selected:
                if args.merge_mode == "top1":
                    selected = [selected[int(np.argmax(scores[selected]))]]
                selected_masks = masks[selected, 0]
                binary_mask = (np.max(selected_masks, axis=0) >= args.mask_threshold).astype(
                    np.uint8
                ) * 255
                metadata["frames_with_detection"] += 1
            else:
                metadata["frames_without_detection"] += 1

            for det_idx in selected:
                frame_summary["detections"].append(
                    {
                        "label": categories[int(labels[det_idx])],
                        "score": float(scores[det_idx]),
                    }
                )

            if selected:
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
        description="Generate TUM RGB-D object masks with torchvision Mask R-CNN."
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
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    generate_masks(parser.parse_args())
