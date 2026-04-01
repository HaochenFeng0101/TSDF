#!/usr/bin/env python3
"""Visualize ScanObjectNN-style h5 point clouds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import h5py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("h5py is required to read h5 files.") from exc

try:
    import open3d as o3d
except Exception:
    o3d = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


SCANOBJECTNN_LABELS = [
    "bag",
    "bin",
    "box",
    "cabinet",
    "chair",
    "desk",
    "display",
    "door",
    "shelf",
    "table",
    "bed",
    "pillow",
    "sink",
    "sofa",
    "toilet",
]


def load_h5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        data = np.asarray(handle["data"], dtype=np.float32)
        labels = np.asarray(handle["label"]).reshape(-1).astype(np.int64)
    return data, labels


def label_name_from_idx(label_idx: int) -> str:
    return SCANOBJECTNN_LABELS[label_idx] if 0 <= label_idx < len(SCANOBJECTNN_LABELS) else str(label_idx)


def resolve_sample_index(labels: np.ndarray, index: int, class_name: str | None, class_offset: int) -> tuple[int, str, int]:
    if class_name is None:
        label_idx = int(labels[index])
        return index, label_name_from_idx(label_idx), label_idx

    class_name = class_name.strip().lower()
    if class_name not in SCANOBJECTNN_LABELS:
        raise ValueError(f"Unknown class '{class_name}'. Choose from {SCANOBJECTNN_LABELS}")

    target_label = SCANOBJECTNN_LABELS.index(class_name)
    matched = np.flatnonzero(labels == target_label)
    if len(matched) == 0:
        raise ValueError(f"No samples with class '{class_name}' were found in the h5 file.")
    if class_offset < 0 or class_offset >= len(matched):
        raise IndexError(
            f"class-offset {class_offset} is out of range for class '{class_name}' with {len(matched)} samples."
        )
    resolved_index = int(matched[class_offset])
    return resolved_index, class_name, target_label


def build_open3d_cloud(points: np.ndarray, color: tuple[float, float, float], offset_x: float):
    cloud = o3d.geometry.PointCloud()
    shifted = points[:, :3].astype(np.float64, copy=True)
    shifted[:, 0] += offset_x
    cloud.points = o3d.utility.Vector3dVector(shifted)
    cloud.paint_uniform_color(color)
    return cloud


def show_with_open3d(primary_points: np.ndarray, compare_points: np.ndarray | None, offset: float, point_size: float) -> None:
    geometries = [build_open3d_cloud(primary_points, color=(0.1, 0.55, 0.95), offset_x=0.0)]
    if compare_points is not None:
        geometries.append(build_open3d_cloud(compare_points, color=(0.95, 0.35, 0.15), offset_x=offset))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="H5 Point Cloud Viewer", width=1440, height=900)
    render_option = vis.get_render_option()
    render_option.point_size = float(point_size)
    render_option.background_color = np.array([0.03, 0.03, 0.03], dtype=np.float64)
    for geom in geometries:
        vis.add_geometry(geom)
    vis.run()
    vis.destroy_window()


def show_with_matplotlib(
    primary_points: np.ndarray,
    compare_points: np.ndarray | None,
    offset: float,
    title: str,
) -> None:
    if plt is None:
        raise RuntimeError("Neither open3d nor matplotlib is available for visualization.")

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="3d")
    p = primary_points[:, :3]
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=5, c="#2b8cfe", alpha=0.85, label="primary")

    if compare_points is not None:
        c = compare_points[:, :3].copy()
        c[:, 0] += offset
        ax.scatter(c[:, 0], c[:, 1], c[:, 2], s=5, c="#f05a28", alpha=0.8, label="compare")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one or two h5 point cloud samples.")
    parser.add_argument("--h5", required=True, help="Primary h5 file.")
    parser.add_argument("--compare-h5", default=None, help="Optional second h5 file for side-by-side comparison.")
    parser.add_argument("--index", type=int, default=0, help="Global sample index to visualize.")
    parser.add_argument("--class-name", default=None, help="Optional class name such as chair, bag, sofa.")
    parser.add_argument(
        "--class-offset",
        type=int,
        default=0,
        help="When class-name is set, choose the N-th sample inside that class.",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=1.6,
        help="X-axis offset used when compare-h5 is provided.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=4.0,
        help="Rendered point size for Open3D.",
    )
    args = parser.parse_args()

    h5_path = Path(args.h5)
    data, labels = load_h5(h5_path)
    if args.index < 0 or args.index >= len(data):
        raise IndexError(f"Index {args.index} is out of range for {h5_path} with {len(data)} samples.")

    resolved_index, label_name, label_idx = resolve_sample_index(
        labels=labels,
        index=args.index,
        class_name=args.class_name,
        class_offset=args.class_offset,
    )
    primary_points = data[resolved_index]
    compare_points = None

    print(f"Primary      : {h5_path}")
    print(f"Global index : {resolved_index}")
    if args.class_name is not None:
        print(f"Class        : {label_name} ({label_idx})")
        print(f"Class offset : {args.class_offset}")
    else:
        print(f"Label        : {label_name} ({label_idx})")

    if args.compare_h5:
        compare_path = Path(args.compare_h5)
        compare_data, compare_labels = load_h5(compare_path)
        if len(compare_data) != len(data):
            raise ValueError("compare-h5 does not have the same number of samples as h5.")
        if int(compare_labels[resolved_index]) != label_idx:
            print("Warning: compare-h5 has a different label at the same global index.")
        compare_points = compare_data[resolved_index]
        compare_label = int(compare_labels[resolved_index])
        compare_name = label_name_from_idx(compare_label)
        print(f"Compare      : {compare_path}")
        print(f"Compare label: {compare_name} ({compare_label})")
        print("Blue = primary, Red = compare-h5 (shifted along +X).")

    title = f"Sample {resolved_index} | {label_name}"
    if o3d is not None:
        show_with_open3d(primary_points, compare_points, offset=args.offset, point_size=args.point_size)
    else:
        print("open3d is not installed, falling back to matplotlib.")
        show_with_matplotlib(primary_points, compare_points, offset=args.offset, title=title)


if __name__ == "__main__":
    main()
