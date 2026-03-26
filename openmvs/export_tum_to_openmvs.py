#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import yaml


TSDF_ROOT = Path(__file__).resolve().parents[1]


def update_recursive(base, patch):
    for key, value in patch.items():
        if isinstance(value, dict):
            current = base.get(key)
            if not isinstance(current, dict):
                current = {}
            base[key] = current
            update_recursive(current, value)
        else:
            base[key] = value


def load_config(path):
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config_part = yaml.full_load(handle)

    inherit_from = config_part.get("inherit_from")
    if inherit_from:
        inherit_path = Path(inherit_from)
        if not inherit_path.is_absolute():
            inherit_path = (config_path.parent / inherit_path).resolve()
        inherited = load_config(inherit_path)
    else:
        inherited = {}

    update_recursive(inherited, config_part)
    return inherited


def parse_list(path, skiprows=0):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split())

    if skiprows:
        rows = rows[skiprows:]
    return np.asarray(rows, dtype=np.str_)


def associate_frames(tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
    associations = []
    for image_idx, timestamp in enumerate(tstamp_image):
        depth_idx = np.argmin(np.abs(tstamp_depth - timestamp))
        pose_idx = np.argmin(np.abs(tstamp_pose - timestamp))
        if (
            abs(tstamp_depth[depth_idx] - timestamp) < max_dt
            and abs(tstamp_pose[pose_idx] - timestamp) < max_dt
        ):
            associations.append((image_idx, depth_idx, pose_idx))
    return associations


def quaternion_xyzw_to_rotation_matrix(quaternion):
    x, y, z, w = quaternion
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def load_tum_frames(dataset_path, frame_stride=1, max_frames=None, max_dt=0.08):
    dataset_path = Path(dataset_path)
    pose_path = dataset_path / "groundtruth.txt"
    if not pose_path.is_file():
        pose_path = dataset_path / "pose.txt"
    if not pose_path.is_file():
        raise FileNotFoundError(f"Could not find groundtruth.txt or pose.txt in {dataset_path}")

    image_path = dataset_path / "rgb.txt"
    depth_path = dataset_path / "depth.txt"
    if not image_path.is_file() or not depth_path.is_file():
        raise FileNotFoundError(f"Could not find rgb.txt or depth.txt in {dataset_path}")

    image_data = parse_list(image_path)
    depth_data = parse_list(depth_path)
    pose_data = parse_list(pose_path)

    tstamp_image = image_data[:, 0].astype(np.float64)
    tstamp_depth = depth_data[:, 0].astype(np.float64)
    tstamp_pose = pose_data[:, 0].astype(np.float64)
    pose_vecs = pose_data[:, :].astype(np.float64)

    associations = associate_frames(tstamp_image, tstamp_depth, tstamp_pose, max_dt=max_dt)
    if frame_stride > 1:
        associations = associations[::frame_stride]
    if max_frames is not None:
        associations = associations[:max_frames]

    frames = []
    for image_idx, depth_idx, pose_idx in associations:
        translation = pose_vecs[pose_idx][1:4]
        quaternion_xyzw = pose_vecs[pose_idx][4:]
        cam_to_world = np.eye(4, dtype=np.float64)
        cam_to_world[:3, :3] = quaternion_xyzw_to_rotation_matrix(quaternion_xyzw)
        cam_to_world[:3, 3] = translation
        world_to_cam = np.linalg.inv(cam_to_world)

        frames.append(
            {
                "color_path": dataset_path / image_data[image_idx, 1],
                "depth_path": dataset_path / depth_data[depth_idx, 1],
                "extrinsic": world_to_cam,
            }
        )
    return frames


def build_camera_model(config, input_width=None, input_height=None):
    calibration = config["Dataset"]["Calibration"]
    raw_width = calibration["width"]
    raw_height = calibration["height"]
    target_width = input_width or config["Dataset"].get("input_width", raw_width)
    target_height = input_height or config["Dataset"].get("input_height", raw_height)

    scale_x = target_width / raw_width
    scale_y = target_height / raw_height

    distorted = bool(calibration.get("distorted", False))
    if distorted:
        raise ValueError(
            "This lightweight exporter currently supports only undistorted configs. "
            "The provided config has Dataset.Calibration.distorted=True."
        )

    return {
        "raw_width": raw_width,
        "raw_height": raw_height,
        "width": int(target_width),
        "height": int(target_height),
        "fx": float(calibration["fx"] * scale_x),
        "fy": float(calibration["fy"] * scale_y),
        "cx": float(calibration["cx"] * scale_x),
        "cy": float(calibration["cy"] * scale_y),
        "depth_scale": float(calibration["depth_scale"]),
    }


def resize_image_if_needed(image_array, width, height, resample):
    if image_array.shape[1] == width and image_array.shape[0] == height:
        return image_array
    image = Image.fromarray(image_array)
    return np.array(image.resize((width, height), resample=resample))


def rotation_matrix_to_colmap_qvec(rotation):
    trace = np.trace(rotation)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s

    qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
    qvec /= np.linalg.norm(qvec)
    return qvec


def resolve_dataset_path(config, dataset_override):
    if dataset_override:
        dataset_path = Path(dataset_override)
    else:
        dataset_path = Path(config["Dataset"]["dataset_path"])
        if not dataset_path.is_absolute():
            dataset_path = TSDF_ROOT / dataset_path
    return dataset_path.resolve()


def scene_name_from_config(config_path):
    return Path(config_path).stem


def default_workspace_name(config_path):
    return f"{scene_name_from_config(config_path)}_openmvs"


def write_cameras_txt(path, camera):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Camera list with one line of data per camera:\n")
        handle.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        handle.write("# Number of cameras: 1\n")
        handle.write(
            "1 PINHOLE "
            f"{camera['width']} {camera['height']} "
            f"{camera['fx']:.8f} {camera['fy']:.8f} {camera['cx']:.8f} {camera['cy']:.8f}\n"
        )


def write_images_txt(path, image_entries):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        handle.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        handle.write(f"# Number of images: {len(image_entries)}\n")
        for entry in image_entries:
            qvec = entry["qvec"]
            tvec = entry["tvec"]
            handle.write(
                f"{entry['image_id']} "
                f"{qvec[0]:.12f} {qvec[1]:.12f} {qvec[2]:.12f} {qvec[3]:.12f} "
                f"{tvec[0]:.12f} {tvec[1]:.12f} {tvec[2]:.12f} "
                f"1 {entry['image_name']}\n"
            )
            handle.write("\n")


def write_points3d_txt(path):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# 3D point list with one line of data per point:\n")
        handle.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        handle.write("# Number of points: 0\n")


def save_seed_point_cloud(
    frames,
    camera,
    output_path,
    sample_frame_stride,
    sample_pixel_stride,
    depth_min,
    depth_max,
):
    points_world = []
    colors = []

    us = np.arange(0, camera["width"], sample_pixel_stride)
    vs = np.arange(0, camera["height"], sample_pixel_stride)
    grid_u, grid_v = np.meshgrid(us, vs)

    sampled_frames = frames[::sample_frame_stride]
    total_frames = len(sampled_frames)
    for frame_idx, frame in enumerate(sampled_frames, start=1):
        color = np.array(Image.open(frame["color_path"]).convert("RGB"))
        depth = np.array(Image.open(frame["depth_path"]))

        color = resize_image_if_needed(color, camera["width"], camera["height"], Image.BILINEAR)
        depth = resize_image_if_needed(depth, camera["width"], camera["height"], Image.NEAREST)

        sampled_depth = depth[grid_v, grid_u].astype(np.float32) / camera["depth_scale"]
        valid = np.isfinite(sampled_depth)
        valid &= sampled_depth > depth_min
        valid &= sampled_depth < depth_max
        if not np.any(valid):
            continue

        z = sampled_depth[valid]
        u = grid_u[valid].astype(np.float32)
        v = grid_v[valid].astype(np.float32)
        x = (u - camera["cx"]) * z / camera["fx"]
        y = (v - camera["cy"]) * z / camera["fy"]
        cam_points = np.stack([x, y, z], axis=1)

        cam_to_world = np.linalg.inv(frame["extrinsic"])
        rotation = cam_to_world[:3, :3]
        translation = cam_to_world[:3, 3]
        world_points = (rotation @ cam_points.T).T + translation

        sampled_colors = color[grid_v[valid], grid_u[valid], :3]
        points_world.append(world_points)
        colors.append(sampled_colors)
        print(f"Seed cloud sampling: processed frame {frame_idx}/{total_frames}")

    if not points_world:
        raise RuntimeError("No valid depth samples were found for the seed point cloud.")

    points_world = np.concatenate(points_world, axis=0)
    colors = np.concatenate(colors, axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points_world)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points_world, colors):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )

    print(f"Saved seed point cloud to {output_path}")


def export_workspace(args):
    config = load_config(args.config)
    dataset_path = resolve_dataset_path(config, args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    workspace_name = args.workspace_name or default_workspace_name(args.config)
    workspace_dir = (TSDF_ROOT / "openmvs" / "workspaces" / workspace_name).resolve()
    if workspace_dir.exists() and not args.no_clean_workspace:
        print(f"Cleaning existing workspace: {workspace_dir}")
        shutil.rmtree(workspace_dir)
    images_dir = workspace_dir / "images"
    sparse_dir = workspace_dir / "colmap" / "sparse"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    frames = load_tum_frames(
        dataset_path,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        max_dt=args.max_dt,
    )
    if not frames:
        raise RuntimeError("No synchronized TUM RGB-D frames were found.")

    camera = build_camera_model(config, args.input_width, args.input_height)
    image_entries = []

    for image_id, frame in enumerate(frames, start=1):
        image_name = f"{image_id:06d}.png"
        image_path = images_dir / image_name

        color = np.array(Image.open(frame["color_path"]).convert("RGB"))
        color = resize_image_if_needed(color, camera["width"], camera["height"], Image.BILINEAR)
        Image.fromarray(color.astype(np.uint8)).save(image_path)

        rotation = frame["extrinsic"][:3, :3]
        translation = frame["extrinsic"][:3, 3]
        image_entries.append(
            {
                "image_id": image_id,
                "image_name": image_name,
                "qvec": rotation_matrix_to_colmap_qvec(rotation),
                "tvec": translation.astype(np.float64),
            }
        )

        if image_id % args.log_every == 0 or image_id == len(frames):
            print(f"Exported {image_id}/{len(frames)} images")

    write_cameras_txt(sparse_dir / "cameras.txt", camera)
    write_images_txt(sparse_dir / "images.txt", image_entries)
    write_points3d_txt(sparse_dir / "points3D.txt")

    metadata_path = workspace_dir / "workspace_info.txt"
    with metadata_path.open("w", encoding="utf-8") as handle:
        handle.write(f"config={Path(args.config).resolve()}\n")
        handle.write(f"dataset={dataset_path}\n")
        handle.write(f"frames={len(frames)}\n")
        handle.write(f"frame_stride={args.frame_stride}\n")
        handle.write(f"width={camera['width']}\n")
        handle.write(f"height={camera['height']}\n")

    if not args.skip_seed_cloud:
        save_seed_point_cloud(
            frames=frames,
            camera=camera,
            output_path=workspace_dir / "seed_from_depth.ply",
            sample_frame_stride=args.seed_frame_stride,
            sample_pixel_stride=args.seed_pixel_stride,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
        )

    print(f"OpenMVS workspace ready: {workspace_dir}")
    print(f"Images: {images_dir}")
    print(f"COLMAP sparse model: {sparse_dir}")


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Export a TUM RGB-D sequence to an OpenMVS-friendly workspace."
    )
    parser.add_argument(
        "--config",
        default=str(TSDF_ROOT / "configs" / "rgbd" / "tum" / "fr3_office.yaml"),
        help="Path to the TUM config file.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional override for the dataset path.",
    )
    parser.add_argument(
        "--workspace-name",
        default=None,
        help="Workspace folder name under openmvs/workspaces. Defaults to <config_stem>_openmvs.",
    )
    parser.add_argument(
        "--no-clean-workspace",
        action="store_true",
        help="Keep an existing workspace instead of deleting it before export.",
    )
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--max-dt", type=float, default=0.08)
    parser.add_argument("--input-width", type=int, default=None)
    parser.add_argument("--input-height", type=int, default=None)
    parser.add_argument("--depth-min", type=float, default=0.2)
    parser.add_argument("--depth-max", type=float, default=4.0)
    parser.add_argument("--seed-frame-stride", type=int, default=2)
    parser.add_argument("--seed-pixel-stride", type=int, default=16)
    parser.add_argument("--skip-seed-cloud", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    export_workspace(parser.parse_args())
