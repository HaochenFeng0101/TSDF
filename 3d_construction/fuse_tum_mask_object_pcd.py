import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh

from run_tum_rgbd_tsdf import (
    TSDF_ROOT,
    associate_frames,
    build_camera_model,
    load_config,
    parse_list,
    resize_or_rectify_color,
    resize_or_rectify_depth,
)


# python 3d_construction/fuse_tum_mask_object_pcd.py \
#   --config configs/rgbd/tum/fr3_office.yaml \
#   --mask-dir mask_generation/outputs/fr3_office_chair/masks \
#   --mask-list mask_generation/outputs/fr3_office_chair/mask.txt \
#   --largest-component \
#   --voxel-downsample 0.005 \
#   --remove-statistical-outlier

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")


def default_output_path(config_path):
    output_dir = TSDF_ROOT / "3d_construction" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{Path(config_path).stem}_masked_object.pcd"


def find_pose_list(dataset_path):
    for filename in ("groundtruth.txt", "pose.txt"):
        candidate = os.path.join(dataset_path, filename)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find groundtruth.txt or pose.txt in {dataset_path}"
    )


def load_mask_table(mask_list_path):
    data = parse_list(mask_list_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    timestamps = data[:, 0].astype(np.float64)
    relpaths = data[:, 1]
    return timestamps, relpaths


def resolve_mask_list_path(mask_dir, mask_relpath):
    mask_relpath = Path(str(mask_relpath))
    if mask_relpath.is_absolute():
        return mask_relpath

    direct = Path(mask_dir) / mask_relpath
    if direct.is_file():
        return direct

    if mask_relpath.parts and mask_relpath.parts[0] == "masks":
        from_parent = Path(mask_dir).parent / mask_relpath
        if from_parent.is_file():
            return from_parent

        if Path(mask_dir).name == "masks":
            trimmed = Path(*mask_relpath.parts[1:])
            trimmed_candidate = Path(mask_dir) / trimmed
            if trimmed_candidate.is_file():
                return trimmed_candidate

    return direct


def resolve_mask_path(mask_dir, rgb_relpath, rgb_timestamp, mask_suffix):
    rgb_relpath = str(rgb_relpath)
    rgb_path = Path(rgb_relpath)
    stem = rgb_path.stem
    suffix = rgb_path.suffix or ".png"
    timestamp_str = str(rgb_timestamp)

    candidates = [
        Path(mask_dir) / rgb_relpath,
        Path(mask_dir) / rgb_path.name,
        Path(mask_dir) / f"{stem}{mask_suffix}{suffix}",
        Path(mask_dir) / rgb_path.parent / f"{stem}{mask_suffix}{suffix}",
    ]

    for ext in IMAGE_EXTS:
        candidates.extend(
            [
                Path(mask_dir) / f"{stem}{mask_suffix}{ext}",
                Path(mask_dir) / rgb_path.parent / f"{stem}{mask_suffix}{ext}",
                Path(mask_dir) / f"{timestamp_str}{mask_suffix}{ext}",
                Path(mask_dir) / rgb_path.parent / f"{timestamp_str}{mask_suffix}{ext}",
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_tum_masked_frames(
    dataset_path,
    mask_dir,
    mask_list=None,
    mask_suffix="",
    frame_stride=1,
    max_frames=None,
    max_dt=0.08,
):
    pose_list = find_pose_list(dataset_path)
    image_list = os.path.join(dataset_path, "rgb.txt")
    depth_list = os.path.join(dataset_path, "depth.txt")
    if not os.path.isfile(image_list) or not os.path.isfile(depth_list):
        raise FileNotFoundError(
            f"Could not find rgb.txt or depth.txt in {dataset_path}"
        )

    image_data = parse_list(image_list)
    depth_data = parse_list(depth_list)
    pose_data = parse_list(pose_list, skiprows=1)

    tstamp_image = image_data[:, 0].astype(np.float64)
    tstamp_depth = depth_data[:, 0].astype(np.float64)
    tstamp_pose = pose_data[:, 0].astype(np.float64)
    pose_vecs = pose_data[:, :].astype(np.float64)

    if mask_list is not None:
        tstamp_mask, mask_relpaths = load_mask_table(mask_list)
    else:
        tstamp_mask, mask_relpaths = None, None

    associations = associate_frames(tstamp_image, tstamp_depth, tstamp_pose, max_dt)
    if frame_stride > 1:
        associations = associations[::frame_stride]
    if max_frames is not None:
        associations = associations[:max_frames]

    frames = []
    missing_masks = 0
    for image_idx, depth_idx, pose_idx in associations:
        rgb_relpath = image_data[image_idx, 1]
        rgb_timestamp = image_data[image_idx, 0]

        if tstamp_mask is not None:
            mask_idx = np.argmin(np.abs(tstamp_mask - float(rgb_timestamp)))
            if np.abs(tstamp_mask[mask_idx] - float(rgb_timestamp)) >= max_dt:
                missing_masks += 1
                continue
            mask_path = resolve_mask_list_path(mask_dir, mask_relpaths[mask_idx])
        else:
            mask_path = resolve_mask_path(mask_dir, rgb_relpath, rgb_timestamp, mask_suffix)

        if mask_path is None or not Path(mask_path).is_file():
            missing_masks += 1
            continue

        quat = pose_vecs[pose_idx][4:]
        trans = pose_vecs[pose_idx][1:4]
        cam_to_world = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
        cam_to_world[:3, 3] = trans

        frames.append(
            {
                "timestamp": float(rgb_timestamp),
                "color_path": os.path.join(dataset_path, rgb_relpath),
                "depth_path": os.path.join(dataset_path, depth_data[depth_idx, 1]),
                "mask_path": str(mask_path),
                "cam_to_world": cam_to_world,
            }
        )
    return frames, missing_masks


def load_mask(mask_path):
    mask_path = Path(mask_path)
    if mask_path.suffix.lower() == ".npy":
        mask = np.load(mask_path)
    else:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask file: {mask_path}")

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def preprocess_mask(mask, width, height, threshold, erode_kernel, dilate_kernel, largest_component):
    if mask.shape[1] != width or mask.shape[0] != height:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    if mask.dtype == np.bool_:
        mask_binary = mask
    else:
        mask_binary = mask.astype(np.float32) > threshold

    mask_binary = mask_binary.astype(np.uint8)

    if erode_kernel > 0:
        kernel = np.ones((erode_kernel, erode_kernel), dtype=np.uint8)
        mask_binary = cv2.erode(mask_binary, kernel, iterations=1)
    if dilate_kernel > 0:
        kernel = np.ones((dilate_kernel, dilate_kernel), dtype=np.uint8)
        mask_binary = cv2.dilate(mask_binary, kernel, iterations=1)

    if largest_component:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, 8)
        if num_labels > 1:
            component_areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = 1 + int(np.argmax(component_areas))
            mask_binary = labels == largest_idx
        else:
            mask_binary = mask_binary.astype(bool)
    else:
        mask_binary = mask_binary.astype(bool)
    return mask_binary


def masked_depth_to_world_points(color, depth, mask, camera, cam_to_world, depth_min, depth_trunc):
    valid_depth = depth.astype(np.float32) / camera["depth_scale"]
    valid = mask & (valid_depth > depth_min) & (valid_depth < depth_trunc)
    if not np.any(valid):
        return None, None

    v, u = np.nonzero(valid)
    z = valid_depth[v, u]
    x = (u.astype(np.float32) - camera["cx"]) * z / camera["fx"]
    y = (v.astype(np.float32) - camera["cy"]) * z / camera["fy"]

    points_cam = np.stack([x, y, z], axis=1)
    points_world = (
        points_cam @ cam_to_world[:3, :3].T + cam_to_world[:3, 3][None, :]
    )
    colors = color[v, u].astype(np.float32) / 255.0
    return points_world, colors


def fuse_masked_object(args):
    config = load_config(args.config)
    dataset_path = args.dataset or config["Dataset"]["dataset_path"]

    frames, missing_masks = load_tum_masked_frames(
        dataset_path=dataset_path,
        mask_dir=args.mask_dir,
        mask_list=args.mask_list,
        mask_suffix=args.mask_suffix,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        max_dt=args.max_dt,
    )
    if not frames:
        raise RuntimeError(
            "No synchronized RGB-D-mask frames were found. Check mask paths or mask timestamps."
        )

    camera = build_camera_model(config, args.input_width, args.input_height)
    all_points = []
    all_colors = []
    used_frames = 0
    empty_mask_frames = 0

    print(
        f"Loaded {len(frames)} usable RGB-D-mask frames"
        f" ({missing_masks} RGB frames had no matching mask)."
    )

    for idx, frame in enumerate(frames, start=1):
        color_bgr = cv2.imread(frame["color_path"], cv2.IMREAD_COLOR)
        if color_bgr is None:
            raise FileNotFoundError(f"Could not read color image: {frame['color_path']}")
        color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        depth = np.array(o3d.io.read_image(frame["depth_path"]))
        mask = load_mask(frame["mask_path"])

        color = resize_or_rectify_color(color, camera)
        depth = resize_or_rectify_depth(depth, camera)
        mask = preprocess_mask(
            mask,
            camera["width"],
            camera["height"],
            args.mask_threshold,
            args.erode_kernel,
            args.dilate_kernel,
            args.largest_component,
        )

        if int(mask.sum()) < args.min_mask_pixels:
            empty_mask_frames += 1
            continue

        points_world, colors = masked_depth_to_world_points(
            color=color,
            depth=depth,
            mask=mask,
            camera=camera,
            cam_to_world=frame["cam_to_world"],
            depth_min=args.depth_min,
            depth_trunc=args.depth_trunc,
        )
        if points_world is None:
            empty_mask_frames += 1
            continue

        all_points.append(points_world)
        all_colors.append(colors)
        used_frames += 1

        if idx % args.log_every == 0 or idx == len(frames):
            print(
                f"Processed {idx}/{len(frames)} frames | "
                f"used={used_frames}, skipped_empty={empty_mask_frames}"
            )

    if not all_points:
        raise RuntimeError(
            "No 3D object points were reconstructed from the masks. "
            "Try lowering --min-mask-pixels or check depth/mask alignment."
        )

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    if args.voxel_downsample > 0:
        point_cloud = point_cloud.voxel_down_sample(args.voxel_downsample)
    if args.remove_statistical_outlier:
        point_cloud, _ = point_cloud.remove_statistical_outlier(
            nb_neighbors=args.outlier_nb_neighbors,
            std_ratio=args.outlier_std_ratio,
        )

    output_path = Path(args.output) if args.output else default_output_path(args.config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), point_cloud)

    print(f"Saved fused masked object point cloud to {output_path}")
    print(
        f"Frames used: {used_frames}/{len(frames)} | "
        f"points after filtering: {len(point_cloud.points)}"
    )


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Fuse object point clouds from TUM RGB-D multi-frame masks."
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
        "--mask-dir",
        required=True,
        help="Directory containing masks. Supports mirrored rgb/ paths or flat same-name masks.",
    )
    parser.add_argument(
        "--mask-list",
        default=None,
        help="Optional TUM-style txt file with mask timestamps and relative paths.",
    )
    parser.add_argument(
        "--mask-suffix",
        default="",
        help="Optional suffix before the mask extension, e.g. '_mask'.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .pcd path. Defaults to TSDF/3d_construction/outputs/<config_stem>_masked_object.pcd",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-dt", type=float, default=0.08)
    parser.add_argument("--input-width", type=int, default=None)
    parser.add_argument("--input-height", type=int, default=None)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-mask-pixels", type=int, default=64)
    parser.add_argument("--erode-kernel", type=int, default=0)
    parser.add_argument("--dilate-kernel", type=int, default=0)
    parser.add_argument("--largest-component", action="store_true")
    parser.add_argument("--depth-min", type=float, default=0.05)
    parser.add_argument("--depth-trunc", type=float, default=4.0)
    parser.add_argument("--voxel-downsample", type=float, default=0.005)
    parser.add_argument("--remove-statistical-outlier", action="store_true")
    parser.add_argument("--outlier-nb-neighbors", type=int, default=20)
    parser.add_argument("--outlier-std-ratio", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=50)
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    fuse_masked_object(parser.parse_args())
