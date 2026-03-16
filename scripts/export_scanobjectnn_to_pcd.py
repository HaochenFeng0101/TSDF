#!/usr/bin/env python3
"""Export ScanObjectNN test samples to .pcd files for query with find_and_classify_object_pcd."""

import argparse
from pathlib import Path

import h5py
import numpy as np
import open3d as o3d


SCANOBJECTNN_LABELS = [
    "bag", "bin", "box", "cabinet", "chair", "desk", "display",
    "door", "shelf", "table", "bed", "pillow", "sink", "sofa", "toilet",
]

VARIANT_TO_FILES = {
    "obj_bg": ("training_objectdataset.h5", "test_objectdataset.h5"),
    "obj_only": ("training_objectdataset.h5", "test_objectdataset.h5"),
    "pb_t25": ("training_objectdataset_augmented25_norot.h5", "test_objectdataset_augmented25_norot.h5"),
    "pb_t25_r": ("training_objectdataset_augmented25rot.h5", "test_objectdataset_augmented25rot.h5"),
    "pb_t50_r": ("training_objectdataset_augmentedrot.h5", "test_objectdataset_augmentedrot.h5"),
    "pb_t50_rs": ("training_objectdataset_augmentedrot_scale75.h5", "test_objectdataset_augmentedrot_scale75.h5"),
}


def find_h5_path(root: Path, variant: str, split: str) -> Path:
    """Locate the h5 file for the given variant and split."""
    root = Path(root)
    for subdir in ["main_split", "h5_files/main_split"]:
        filename = VARIANT_TO_FILES[variant][1] if split == "test" else VARIANT_TO_FILES[variant][0]
        p = root / subdir / filename
        if p.exists():
            return p
    candidates = list(root.rglob("*test*.h5")) if split == "test" else list(root.rglob("*train*.h5"))
    if not candidates:
        raise FileNotFoundError(f"No {split} h5 file found under {root}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(
        description="Export ScanObjectNN samples to .pcd for find_and_classify_object_pcd."
    )
    parser.add_argument(
        "--scanobjectnn-root",
        default="data/ScanObjectNN",
        help="Root directory of ScanObjectNN (created by download_scanobjectnn.py).",
    )
    parser.add_argument(
        "--variant",
        default="pb_t50_rs",
        choices=list(VARIANT_TO_FILES),
        help="ScanObjectNN variant.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Train or test split.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/query_samples",
        help="Output directory for .pcd files.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to export (0 = all).",
    )
    parser.add_argument(
        "--label-filter",
        default=None,
        help="Only export samples of this class (e.g. chair).",
    )
    args = parser.parse_args()

    root = Path(args.scanobjectnn_root)
    if not root.exists():
        raise FileNotFoundError(f"ScanObjectNN root not found: {root}")

    h5_path = find_h5_path(root, args.variant, args.split)

    with h5py.File(h5_path, "r") as f:
        data = np.asarray(f["data"])
        labels = np.asarray(f["label"]).flatten()

    indices = list(range(len(data)))
    if args.label_filter:
        label_filter = args.label_filter.strip().lower()
        try:
            label_idx = SCANOBJECTNN_LABELS.index(label_filter)
        except ValueError:
            raise ValueError(
                f"Unknown label '{label_filter}'. Choose from {SCANOBJECTNN_LABELS}"
            )
        indices = [i for i in indices if int(labels[i]) == label_idx]
        if not indices:
            raise ValueError(f"No samples with label '{label_filter}' in {h5_path}")

    if args.num_samples > 0:
        indices = indices[: args.num_samples]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        pts = data[idx][:, :3]
        label_name = SCANOBJECTNN_LABELS[int(labels[idx])]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        out_path = output_dir / f"sample_{idx}_label_{label_name}.pcd"
        o3d.io.write_point_cloud(str(out_path), pcd)
        print(f"Saved {out_path}")

    print(f"Exported {len(indices)} samples to {output_dir}")


if __name__ == "__main__":
    main()
