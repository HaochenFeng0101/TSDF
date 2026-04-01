#!/usr/bin/env python3
"""Apply mild point corruption to ScanObjectNN h5 files.

The script keeps the output compatible with the existing training code:
- dataset keys stay as `data`, `label`, and optional `mask`
- point count per sample is preserved

Per sample, exactly one corruption is applied:
- random global point dropout
- one local missing chunk around a random anchor point
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import h5py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("h5py is required to process ScanObjectNN h5 files.") from exc


VARIANT_TO_FILES = {
    "obj_bg": (
        "training_objectdataset.h5",
        "test_objectdataset.h5",
    ),
    "obj_only": (
        "training_objectdataset.h5",
        "test_objectdataset.h5",
    ),
    "pb_t25": (
        "training_objectdataset_augmented25_norot.h5",
        "test_objectdataset_augmented25_norot.h5",
    ),
    "pb_t25_r": (
        "training_objectdataset_augmented25rot.h5",
        "test_objectdataset_augmented25rot.h5",
    ),
    "pb_t50_r": (
        "training_objectdataset_augmentedrot.h5",
        "test_objectdataset_augmentedrot.h5",
    ),
    "pb_t50_rs": (
        "training_objectdataset_augmentedrot_scale75.h5",
        "test_objectdataset_augmentedrot_scale75.h5",
    ),
}


def resolve_input_h5(root: Path, variant: str, split: str, use_background: bool) -> Path:
    split_dir_name = "main_split" if use_background else "main_split_nobg"
    candidates = [root / split_dir_name, root / "h5_files" / split_dir_name]
    split_dir = next((candidate for candidate in candidates if candidate.exists()), None)
    if split_dir is None:
        raise FileNotFoundError(f"Could not find {split_dir_name} under {root}")

    filename = VARIANT_TO_FILES[variant][0 if split == "train" else 1]
    h5_path = split_dir / filename
    if not h5_path.exists():
        raise FileNotFoundError(f"Could not find ScanObjectNN h5 file: {h5_path}")
    return h5_path


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32, copy=False)
    centroid = points.mean(axis=0, keepdims=True)
    centered = points - centroid
    scale = np.linalg.norm(centered, axis=1).max()
    if scale > 0:
        centered = centered / scale
    return centered


def apply_single_corruption(
    points: np.ndarray,
    reference_points: np.ndarray,
    rng: np.random.Generator,
    random_drop_min: float,
    random_drop_max: float,
    local_drop_min: float,
    local_drop_max: float,
    min_keep_ratio: float,
) -> tuple[np.ndarray, dict[str, float]]:
    num_points = len(points)
    work_points = points.astype(np.float32, copy=True)
    distance_points = reference_points.astype(np.float32, copy=False)

    random_drop_ratio = float(rng.uniform(random_drop_min, random_drop_max))
    local_drop_ratio = float(rng.uniform(local_drop_min, local_drop_max))
    corruption_type = "random_dropout" if rng.random() < 0.5 else "local_dropout"

    keep_mask = np.ones(num_points, dtype=bool)

    if corruption_type == "random_dropout":
        if random_drop_ratio > 0:
            random_drop = rng.random(num_points) < random_drop_ratio
            keep_mask[random_drop] = False
    else:
        current_keep = np.flatnonzero(keep_mask)
        if local_drop_ratio > 0 and len(current_keep) > 8:
            anchor = int(rng.choice(current_keep))
            target_local_drop = max(1, int(round(num_points * local_drop_ratio)))
            local_drop = min(target_local_drop, max(1, len(current_keep) - 8))
            if local_drop > 0:
                distances = np.linalg.norm(distance_points - distance_points[anchor], axis=1)
                nearest = np.argsort(distances)
                local_candidates = nearest[keep_mask[nearest]]
                keep_mask[local_candidates[:local_drop]] = False

    keep_indices = np.flatnonzero(keep_mask)
    min_keep = max(8, int(round(num_points * min_keep_ratio)))
    if len(keep_indices) < min_keep:
        restore_count = min_keep - len(keep_indices)
        restore_candidates = np.flatnonzero(~keep_mask)
        if restore_count > 0 and len(restore_candidates) > 0:
            restore = rng.choice(
                restore_candidates,
                size=min(restore_count, len(restore_candidates)),
                replace=False,
            )
            keep_mask[restore] = True
            keep_indices = np.flatnonzero(keep_mask)

    kept = work_points[keep_indices]
    if len(kept) == 0:
        kept = work_points[:1]

    if len(kept) < num_points:
        duplicate_idx = rng.choice(len(kept), size=num_points - len(kept), replace=True)
        duplicated = kept[duplicate_idx]
        corrupted = np.concatenate([kept, duplicated], axis=0)
    elif len(kept) > num_points:
        sample_idx = rng.choice(len(kept), size=num_points, replace=False)
        corrupted = kept[sample_idx]
    else:
        corrupted = kept

    corrupted = corrupted[rng.permutation(num_points)].astype(np.float32, copy=False)
    stats = {
        "corruption_type": corruption_type,
        "random_drop_ratio": random_drop_ratio if corruption_type == "random_dropout" else 0.0,
        "local_drop_ratio": local_drop_ratio if corruption_type == "local_dropout" else 0.0,
        "kept_ratio_before_refill": float(len(keep_indices) / num_points),
        "final_unique_ratio": float(len(np.unique(corrupted, axis=0)) / num_points),
    }
    return corrupted, stats


def process_dataset(
    data: np.ndarray,
    seed: int,
    random_drop_min: float,
    random_drop_max: float,
    local_drop_min: float,
    local_drop_max: float,
    min_keep_ratio: float,
    normalize_before_chunk: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)
    processed = np.empty_like(data, dtype=np.float32)
    random_drop_values = []
    local_drop_values = []
    keep_values = []
    unique_values = []
    random_mode_count = 0
    local_mode_count = 0

    for idx in range(len(data)):
        points = data[idx][:, :3].astype(np.float32, copy=True)
        reference_points = normalize_points(points) if normalize_before_chunk else points
        corrupted, stats = apply_single_corruption(
            points=points,
            reference_points=reference_points,
            rng=rng,
            random_drop_min=random_drop_min,
            random_drop_max=random_drop_max,
            local_drop_min=local_drop_min,
            local_drop_max=local_drop_max,
            min_keep_ratio=min_keep_ratio,
        )

        if data.shape[-1] > 3:
            processed[idx, :, :3] = corrupted
            processed[idx, :, 3:] = data[idx, :, 3:]
        else:
            processed[idx] = corrupted

        random_drop_values.append(stats["random_drop_ratio"])
        local_drop_values.append(stats["local_drop_ratio"])
        keep_values.append(stats["kept_ratio_before_refill"])
        unique_values.append(stats["final_unique_ratio"])
        if stats["corruption_type"] == "random_dropout":
            random_mode_count += 1
        else:
            local_mode_count += 1

    summary = {
        "mean_random_drop_ratio": float(np.mean(random_drop_values)),
        "mean_local_drop_ratio": float(np.mean(local_drop_values)),
        "mean_kept_ratio_before_refill": float(np.mean(keep_values)),
        "mean_final_unique_ratio": float(np.mean(unique_values)),
        "random_mode_count": float(random_mode_count),
        "local_mode_count": float(local_mode_count),
    }
    return processed, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a mildly corrupted ScanObjectNN h5 file for more realistic training."
    )
    parser.add_argument("--scanobjectnn-root", default="data/ScanObjectNN")
    parser.add_argument("--variant", default="pb_t50_rs", choices=sorted(VARIANT_TO_FILES))
    parser.add_argument("--split", default="train", choices=["train", "test", "both"])
    parser.add_argument("--no-bg", action="store_true", help="Use main_split_nobg.")
    parser.add_argument(
        "--output-root",
        default="data/ScanObjectNN_mild",
        help="Root folder for processed h5 files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-drop-min", type=float, default=0.03)
    parser.add_argument("--random-drop-max", type=float, default=0.08)
    parser.add_argument("--local-drop-min", type=float, default=0.05)
    parser.add_argument("--local-drop-max", type=float, default=0.12)
    parser.add_argument(
        "--min-keep-ratio",
        type=float,
        default=0.82,
        help="Guarantee at least this many original points remain before refill.",
    )
    parser.add_argument(
        "--normalize-before-chunk",
        action="store_true",
        help="Normalize each sample before computing local chunk distance.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output h5 if it already exists.",
    )
    args = parser.parse_args()

    input_root = Path(args.scanobjectnn_root)
    output_root = Path(args.output_root)
    split_dir_name = "main_split" if not args.no_bg else "main_split_nobg"
    output_split_dir = output_root / split_dir_name
    output_split_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "test"] if args.split == "both" else [args.split]

    for split_index, split in enumerate(splits):
        input_h5 = resolve_input_h5(
            root=input_root,
            variant=args.variant,
            split=split,
            use_background=not args.no_bg,
        )
        output_h5 = output_split_dir / input_h5.name
        if output_h5.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output file already exists: {output_h5}. Use --overwrite to replace it."
            )

        with h5py.File(input_h5, "r") as src:
            data = np.asarray(src["data"], dtype=np.float32)
            labels = np.asarray(src["label"])
            mask = np.asarray(src["mask"], dtype=np.float32) if "mask" in src else None

        processed_data, summary = process_dataset(
            data=data,
            seed=args.seed + split_index,
            random_drop_min=args.random_drop_min,
            random_drop_max=args.random_drop_max,
            local_drop_min=args.local_drop_min,
            local_drop_max=args.local_drop_max,
            min_keep_ratio=args.min_keep_ratio,
            normalize_before_chunk=args.normalize_before_chunk,
        )

        with h5py.File(output_h5, "w") as dst:
            dst.create_dataset("data", data=processed_data, compression="gzip")
            dst.create_dataset("label", data=labels, compression="gzip")
            if mask is not None:
                dst.create_dataset("mask", data=mask, compression="gzip")

        print(f"[{split}] input : {input_h5}")
        print(f"[{split}] output: {output_h5}")
        print(
            f"[{split}] mean_random_drop={summary['mean_random_drop_ratio']:.4f} "
            f"mean_local_drop={summary['mean_local_drop_ratio']:.4f} "
            f"mean_keep_before_refill={summary['mean_kept_ratio_before_refill']:.4f} "
            f"mean_final_unique={summary['mean_final_unique_ratio']:.4f}"
        )
        print(
            f"[{split}] random_mode_samples={int(summary['random_mode_count'])} "
            f"local_mode_samples={int(summary['local_mode_count'])}"
        )


if __name__ == "__main__":
    main()
