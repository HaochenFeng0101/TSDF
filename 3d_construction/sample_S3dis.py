import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

'''
use seed to get same output
 python3 3d_construction/sample_S3dis.py \
  --seed 42



'''
TSDF_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = TSDF_ROOT / "data" / "S3DIS_raw" / "Stanford3dDataset_v1.2_Aligned_Version"
DEFAULT_OUTPUT_DIR = TSDF_ROOT / "3d_construction" / "outputs"


def parse_annotation_file(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 6:
                continue
            try:
                values = [float(value) for value in parts[:6]]
            except ValueError:
                continue
            if not np.all(np.isfinite(values)):
                continue
            rows.append(values)

    if not rows:
        raise ValueError(f"{path} contains no usable xyzrgb rows.")
    return np.asarray(rows, dtype=np.float32)


def list_room_dirs(raw_root, areas=None):
    room_dirs = []
    area_filter = set(areas or [])
    for area_dir in sorted(raw_root.iterdir()):
        if not area_dir.is_dir():
            continue
        if area_filter and area_dir.name not in area_filter:
            continue
        for room_dir in sorted(area_dir.iterdir()):
            if (room_dir / "Annotations").is_dir():
                room_dirs.append(room_dir)
    return room_dirs


def load_room_point_cloud(room_dir):
    annotations_dir = room_dir / "Annotations"
    all_points = []
    for txt_path in sorted(annotations_dir.glob("*.txt")):
        all_points.append(parse_annotation_file(txt_path))
    if not all_points:
        raise RuntimeError(f"No usable annotation files found in {annotations_dir}")
    points = np.concatenate(all_points, axis=0)
    xyz = points[:, :3]
    rgb = np.clip(points[:, 3:6] / 255.0, 0.0, 1.0)
    return xyz.astype(np.float32), rgb.astype(np.float32)


def choose_room(room_dirs, seed=None):
    if not room_dirs:
        raise RuntimeError("No S3DIS rooms were found.")
    rng = np.random.default_rng(seed)
    index = int(rng.integers(0, len(room_dirs)))
    return room_dirs[index]


def write_point_cloud(output_path, xyz, rgb, voxel_downsample=0.0):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    point_cloud.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    if voxel_downsample > 0:
        point_cloud = point_cloud.voxel_down_sample(voxel_downsample)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), point_cloud)
    return point_cloud


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Sample a random S3DIS room and export it as a scene point cloud."
    )
    parser.add_argument(
        "--raw-root",
        default=str(DEFAULT_RAW_ROOT),
        help="Root directory of Stanford3dDataset_v1.2_Aligned_Version.",
    )
    parser.add_argument(
        "--areas",
        nargs="+",
        default=None,
        help="Optional list of Areas to sample from, for example Area_1 Area_2 Area_3.",
    )
    parser.add_argument(
        "--room",
        default=None,
        help="Optional explicit room directory. If set, random sampling is skipped.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .pcd path. Defaults to 3d_construction/outputs/<area>_<room>.pcd",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional metadata .json path. Defaults next to the output .pcd.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible room sampling.",
    )
    parser.add_argument(
        "--voxel-downsample",
        type=float,
        default=0.0,
        help="Optional voxel downsampling size before saving.",
    )
    parser.add_argument("--visualize", action="store_true", help="Visualize the sampled scene.")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"S3DIS raw root not found: {raw_root}")

    if args.room is not None:
        room_dir = Path(args.room)
        if not room_dir.exists():
            raise FileNotFoundError(f"Room directory not found: {room_dir}")
    else:
        room_dirs = list_room_dirs(raw_root, args.areas)
        room_dir = choose_room(room_dirs, seed=args.seed)

    xyz, rgb = load_room_point_cloud(room_dir)
    area_name = room_dir.parent.name
    room_name = room_dir.name

    output_path = (
        Path(args.output)
        if args.output
        else DEFAULT_OUTPUT_DIR / f"{area_name}_{room_name}.pcd"
    )
    point_cloud = write_point_cloud(
        output_path,
        xyz,
        rgb,
        voxel_downsample=args.voxel_downsample,
    )

    metadata_path = (
        Path(args.metadata)
        if args.metadata
        else output_path.with_suffix(".json")
    )
    metadata = {
        "raw_root": str(raw_root.resolve()),
        "area": area_name,
        "room": room_name,
        "room_dir": str(room_dir.resolve()),
        "output": str(output_path.resolve()),
        "points_before_downsample": int(len(xyz)),
        "points_saved": int(len(point_cloud.points)),
        "voxel_downsample": float(args.voxel_downsample),
        "seed": args.seed,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Sampled room: {area_name}/{room_name}")
    print(f"Saved point cloud to {output_path}")
    print(f"Saved metadata to {metadata_path}")

    if args.visualize:
        o3d.visualization.draw_geometries(
            [point_cloud],
            window_name=f"S3DIS Sample: {area_name}/{room_name}",
        )


if __name__ == "__main__":
    main()
