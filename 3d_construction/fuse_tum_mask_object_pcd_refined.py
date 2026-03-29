import argparse
import json
import os
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


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")


def default_output_path(config_path):
    output_dir = TSDF_ROOT / "3d_construction" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{Path(config_path).stem}_masked_object_refined.pcd"


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
    data = np.asarray(data)
    if data.size == 0:
        raise RuntimeError(f"Mask list is empty: {mask_list_path}")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise RuntimeError(
            f"Mask list must contain at least two columns: {mask_list_path}"
        )
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
    rgb_path = Path(str(rgb_relpath))
    stem = rgb_path.stem
    suffix = rgb_path.suffix or ".png"
    timestamp_str = str(rgb_timestamp)

    candidates = [
        Path(mask_dir) / rgb_path,
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
        rgb_timestamp = float(image_data[image_idx, 0])

        if tstamp_mask is not None:
            mask_idx = np.argmin(np.abs(tstamp_mask - rgb_timestamp))
            if np.abs(tstamp_mask[mask_idx] - rgb_timestamp) >= max_dt:
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
                "timestamp": rgb_timestamp,
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


def preprocess_mask(mask, width, height, threshold, largest_component, close_kernel):
    if mask.shape[1] != width or mask.shape[0] != height:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    if mask.dtype == np.bool_:
        mask_binary = mask.astype(np.uint8)
    else:
        mask_binary = (mask.astype(np.float32) > threshold).astype(np.uint8)

    if close_kernel > 0:
        kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)

    if largest_component:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, 8)
        if num_labels > 1:
            component_areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = 1 + int(np.argmax(component_areas))
            mask_binary = (labels == largest_idx).astype(np.uint8)

    return mask_binary.astype(bool)


def shrink_mask(mask, border_kernel):
    if border_kernel <= 0:
        return mask
    kernel = np.ones((border_kernel, border_kernel), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return eroded.astype(bool)


def filter_depth_by_local_consistency(depth_m, valid_mask, kernel_size, threshold_m):
    if kernel_size <= 1 or threshold_m <= 0:
        return valid_mask

    kernel_size = max(3, kernel_size | 1)
    filtered = cv2.medianBlur(depth_m.astype(np.float32), kernel_size)
    depth_delta = np.abs(depth_m - filtered)
    consistent = (filtered > 0) & (depth_delta <= threshold_m)
    return valid_mask & consistent


def masked_depth_to_world_points(color, depth, mask, camera, cam_to_world, args):
    depth_m = depth.astype(np.float32) / camera["depth_scale"]
    valid = (
        mask
        & (depth_m > args.depth_min)
        & (depth_m < args.depth_trunc)
        & np.isfinite(depth_m)
    )
    valid = filter_depth_by_local_consistency(
        depth_m,
        valid,
        args.depth_consistency_kernel,
        args.depth_consistency_threshold,
    )
    if not np.any(valid):
        return None

    v, u = np.nonzero(valid)
    z = depth_m[v, u]
    x = (u.astype(np.float32) - camera["cx"]) * z / camera["fx"]
    y = (v.astype(np.float32) - camera["cy"]) * z / camera["fy"]

    points_cam = np.stack([x, y, z], axis=1)
    points_world = points_cam @ cam_to_world[:3, :3].T + cam_to_world[:3, 3][None, :]
    colors = color[v, u].astype(np.float32) / 255.0
    return build_point_cloud(points_world, colors)


def build_point_cloud(points, colors):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return cloud


def clean_frame_cloud(cloud, args):
    if len(cloud.points) == 0:
        return cloud, {"frame_points_after_cluster": 0, "frame_clusters": 0}

    if args.frame_voxel_size > 0:
        cloud = cloud.voxel_down_sample(args.frame_voxel_size)
    if len(cloud.points) == 0:
        return cloud, {"frame_points_after_cluster": 0, "frame_clusters": 0}

    if args.frame_remove_radius_outlier:
        cloud, _ = cloud.remove_radius_outlier(
            nb_points=args.frame_radius_min_points,
            radius=args.frame_radius_radius,
        )
    if len(cloud.points) == 0:
        return cloud, {"frame_points_after_cluster": 0, "frame_clusters": 0}

    if args.frame_dbscan_eps <= 0 or args.frame_dbscan_min_points <= 1:
        return cloud, {"frame_points_after_cluster": len(cloud.points), "frame_clusters": 1}

    labels = np.asarray(
        cloud.cluster_dbscan(
            eps=args.frame_dbscan_eps,
            min_points=args.frame_dbscan_min_points,
            print_progress=False,
        )
    )
    valid = labels >= 0
    if not np.any(valid):
        return o3d.geometry.PointCloud(), {"frame_points_after_cluster": 0, "frame_clusters": 0}

    cluster_ids, counts = np.unique(labels[valid], return_counts=True)
    cluster_points = np.asarray(cloud.points)

    if args.frame_cluster_mode == "nearest":
        best_cluster = None
        best_depth = None
        for cluster_id in cluster_ids:
            idx = np.where(labels == cluster_id)[0]
            centroid = cluster_points[idx].mean(axis=0)
            depth = float(np.linalg.norm(centroid))
            if best_depth is None or depth < best_depth:
                best_depth = depth
                best_cluster = cluster_id
        keep_cluster_ids = [best_cluster]
    elif args.frame_cluster_mode == "largest":
        best_cluster = int(cluster_ids[int(np.argmax(counts))])
        keep_cluster_ids = [best_cluster]
    else:
        largest_count = int(np.max(counts))
        min_keep = max(
            args.frame_keep_min_cluster_points,
            int(np.ceil(largest_count * args.frame_keep_cluster_ratio)),
        )
        keep_cluster_ids = [
            int(cluster_id)
            for cluster_id, count in zip(cluster_ids, counts)
            if int(count) >= min_keep
        ]

    keep_mask = np.isin(labels, keep_cluster_ids)
    keep_idx = np.where(keep_mask)[0]
    cleaned = cloud.select_by_index(keep_idx)
    return cleaned, {
        "frame_points_after_cluster": len(cleaned.points),
        "frame_clusters": int(len(cluster_ids)),
        "frame_clusters_kept": int(len(keep_cluster_ids)),
    }


def clean_global_cloud(cloud, args):
    stats = {
        "global_points_before_clean": int(len(cloud.points)),
        "global_points_after_clean": 0,
        "global_clusters": 0,
    }
    if len(cloud.points) == 0:
        return cloud, stats

    if args.global_voxel_size > 0:
        cloud = cloud.voxel_down_sample(args.global_voxel_size)
    if len(cloud.points) == 0:
        return cloud, stats

    if args.remove_statistical_outlier:
        cloud, _ = cloud.remove_statistical_outlier(
            nb_neighbors=args.outlier_nb_neighbors,
            std_ratio=args.outlier_std_ratio,
        )
    if len(cloud.points) == 0:
        return cloud, stats

    if args.global_remove_radius_outlier:
        cloud, _ = cloud.remove_radius_outlier(
            nb_points=args.global_radius_min_points,
            radius=args.global_radius_radius,
        )
    if len(cloud.points) == 0:
        return cloud, stats

    if args.global_dbscan_eps > 0 and args.global_dbscan_min_points > 1:
        labels = np.asarray(
            cloud.cluster_dbscan(
                eps=args.global_dbscan_eps,
                min_points=args.global_dbscan_min_points,
                print_progress=False,
            )
        )
        valid = labels >= 0
        if np.any(valid):
            cluster_ids, counts = np.unique(labels[valid], return_counts=True)
            if args.global_cluster_mode == "largest":
                keep_cluster_ids = [int(cluster_ids[int(np.argmax(counts))])]
            else:
                largest_count = int(np.max(counts))
                min_keep = max(
                    args.global_keep_min_cluster_points,
                    int(np.ceil(largest_count * args.global_keep_cluster_ratio)),
                )
                keep_cluster_ids = [
                    int(cluster_id)
                    for cluster_id, count in zip(cluster_ids, counts)
                    if int(count) >= min_keep
                ]
            keep_idx = np.where(np.isin(labels, keep_cluster_ids))[0]
            cloud = cloud.select_by_index(keep_idx)
            stats["global_clusters"] = int(len(cluster_ids))
            stats["global_clusters_kept"] = int(len(keep_cluster_ids))

    stats["global_points_after_clean"] = int(len(cloud.points))
    return cloud, stats


def write_report(output_path, payload):
    report_path = Path(output_path).with_suffix(".refined_report.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved refined fusion report to {report_path}")


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
        raise RuntimeError("No synchronized RGB-D-mask frames were found.")

    camera = build_camera_model(config, args.input_width, args.input_height)
    fused_cloud = o3d.geometry.PointCloud()
    used_frames = 0
    skipped_empty = 0
    skipped_cluster = 0
    frame_reports = []

    print(
        f"Loaded {len(frames)} usable RGB-D-mask frames "
        f"({missing_masks} RGB frames had no matching mask)."
    )

    for idx, frame in enumerate(frames, start=1):
        color_bgr = cv2.imread(frame["color_path"], cv2.IMREAD_COLOR)
        if color_bgr is None:
            raise FileNotFoundError(f"Could not read color image: {frame['color_path']}")
        depth = np.array(o3d.io.read_image(frame["depth_path"]))
        color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        color = resize_or_rectify_color(color, camera)
        depth = resize_or_rectify_depth(depth, camera)

        mask = preprocess_mask(
            load_mask(frame["mask_path"]),
            camera["width"],
            camera["height"],
            args.mask_threshold,
            args.largest_component,
            args.mask_close_kernel,
        )
        mask = shrink_mask(mask, args.mask_border_erode)
        mask_pixels = int(mask.sum())

        frame_report = {
            "timestamp": float(frame["timestamp"]),
            "mask_pixels": mask_pixels,
        }

        if mask_pixels < args.min_mask_pixels:
            skipped_empty += 1
            frame_report["status"] = "skipped_small_mask"
            frame_reports.append(frame_report)
            continue

        frame_cloud = masked_depth_to_world_points(
            color=color,
            depth=depth,
            mask=mask,
            camera=camera,
            cam_to_world=frame["cam_to_world"],
            args=args,
        )
        if frame_cloud is None or len(frame_cloud.points) == 0:
            skipped_empty += 1
            frame_report["status"] = "skipped_no_valid_depth"
            frame_reports.append(frame_report)
            continue

        frame_report["frame_points_raw"] = int(len(frame_cloud.points))
        frame_cloud, clean_stats = clean_frame_cloud(frame_cloud, args)
        frame_report.update(clean_stats)

        if len(frame_cloud.points) < args.min_frame_points_after_clean:
            skipped_cluster += 1
            frame_report["status"] = "skipped_after_frame_clean"
            frame_reports.append(frame_report)
            continue

        fused_cloud += frame_cloud
        used_frames += 1
        frame_report["status"] = "used"
        frame_reports.append(frame_report)

        if idx % args.log_every == 0 or idx == len(frames):
            print(
                f"Processed {idx}/{len(frames)} frames | "
                f"used={used_frames}, skipped_empty={skipped_empty}, "
                f"skipped_cluster={skipped_cluster}"
            )

    if len(fused_cloud.points) == 0:
        raise RuntimeError("No object points remained after refined frame cleaning.")

    fused_cloud, global_stats = clean_global_cloud(fused_cloud, args)
    if len(fused_cloud.points) == 0:
        raise RuntimeError("No object points remained after refined global cleaning.")

    output_path = Path(args.output) if args.output else default_output_path(args.config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), fused_cloud)

    report = {
        "config": str(args.config),
        "dataset": str(dataset_path),
        "mask_dir": str(args.mask_dir),
        "mask_list": str(args.mask_list) if args.mask_list else None,
        "frames_loaded": int(len(frames)),
        "missing_masks": int(missing_masks),
        "frames_used": int(used_frames),
        "frames_skipped_empty": int(skipped_empty),
        "frames_skipped_after_frame_clean": int(skipped_cluster),
        "points_final": int(len(fused_cloud.points)),
        "settings": vars(args),
        "global_stats": global_stats,
        "frame_reports": frame_reports,
    }
    write_report(output_path, report)

    print(f"Saved refined masked object point cloud to {output_path}")
    print(
        f"Frames used: {used_frames}/{len(frames)} | "
        f"points after refined cleaning: {len(fused_cloud.points)}"
    )


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Refined TUM RGB-D object fusion with mask, depth, and 3D cluster cleaning."
    )
    parser.add_argument(
        "--config",
        default=str(TSDF_ROOT / "configs" / "rgbd" / "tum" / "fr3_office.yaml"),
        help="Path to a TUM RGB-D config file.",
    )
    parser.add_argument("--dataset", default=None, help="Override dataset path from the config.")
    parser.add_argument("--mask-dir", required=True, help="Directory containing masks.")
    parser.add_argument(
        "--mask-list",
        default=None,
        help="Optional TUM-style txt file with mask timestamps and relative paths.",
    )
    parser.add_argument("--mask-suffix", default="", help="Optional suffix before mask extension.")
    parser.add_argument("--output", default=None, help="Output .pcd path.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-dt", type=float, default=0.03)
    parser.add_argument("--input-width", type=int, default=None)
    parser.add_argument("--input-height", type=int, default=None)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-mask-pixels", type=int, default=96)
    parser.add_argument("--largest-component", action="store_true")
    parser.add_argument(
        "--mask-close-kernel",
        type=int,
        default=3,
        help="Morphological close kernel before shrink; set 0 to disable.",
    )
    parser.add_argument(
        "--mask-border-erode",
        type=int,
        default=3,
        help="Shrink the mask border to suppress edge-depth leakage.",
    )
    parser.add_argument("--depth-min", type=float, default=0.05)
    parser.add_argument("--depth-trunc", type=float, default=4.0)
    parser.add_argument(
        "--depth-consistency-kernel",
        type=int,
        default=5,
        help="Median filter kernel for local depth consistency checks.",
    )
    parser.add_argument(
        "--depth-consistency-threshold",
        type=float,
        default=0.04,
        help="Reject pixels whose depth differs from local median by more than this many meters.",
    )
    parser.add_argument(
        "--frame-voxel-size",
        type=float,
        default=0.01,
        help="Per-frame voxel size before clustering.",
    )
    parser.add_argument("--frame-remove-radius-outlier", action="store_true")
    parser.add_argument("--frame-radius-min-points", type=int, default=6)
    parser.add_argument("--frame-radius-radius", type=float, default=0.03)
    parser.add_argument(
        "--frame-dbscan-eps",
        type=float,
        default=0.05,
        help="DBSCAN eps for per-frame cluster cleanup. Set 0 to disable.",
    )
    parser.add_argument("--frame-dbscan-min-points", type=int, default=20)
    parser.add_argument(
        "--frame-cluster-mode",
        choices=("largest", "nearest", "multi"),
        default="multi",
        help="Keep the largest, nearest, or multiple sufficiently large per-frame clusters.",
    )
    parser.add_argument(
        "--frame-keep-cluster-ratio",
        type=float,
        default=0.18,
        help="In multi mode, keep per-frame clusters whose size is at least this fraction of the largest cluster.",
    )
    parser.add_argument(
        "--frame-keep-min-cluster-points",
        type=int,
        default=120,
        help="In multi mode, always keep per-frame clusters above this many points after voxelization.",
    )
    parser.add_argument(
        "--min-frame-points-after-clean",
        type=int,
        default=60,
        help="Discard frames that still look too sparse after frame cleanup.",
    )
    parser.add_argument(
        "--global-voxel-size",
        type=float,
        default=0.005,
        help="Final global voxel size before global cleanup.",
    )
    parser.add_argument("--remove-statistical-outlier", action="store_true")
    parser.add_argument("--outlier-nb-neighbors", type=int, default=20)
    parser.add_argument("--outlier-std-ratio", type=float, default=1.8)
    parser.add_argument("--global-remove-radius-outlier", action="store_true")
    parser.add_argument("--global-radius-min-points", type=int, default=10)
    parser.add_argument("--global-radius-radius", type=float, default=0.04)
    parser.add_argument(
        "--global-dbscan-eps",
        type=float,
        default=0.06,
        help="DBSCAN eps for final fused cloud cleanup. Set 0 to disable.",
    )
    parser.add_argument("--global-dbscan-min-points", type=int, default=40)
    parser.add_argument(
        "--global-cluster-mode",
        choices=("largest", "multi"),
        default="multi",
        help="Keep the largest or multiple sufficiently large clusters after global fusion.",
    )
    parser.add_argument(
        "--global-keep-cluster-ratio",
        type=float,
        default=0.12,
        help="In global multi mode, keep clusters whose size is at least this fraction of the largest cluster.",
    )
    parser.add_argument(
        "--global-keep-min-cluster-points",
        type=int,
        default=600,
        help="In global multi mode, always keep clusters above this many points.",
    )
    parser.add_argument("--log-every", type=int, default=50)
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    fuse_masked_object(parser.parse_args())
