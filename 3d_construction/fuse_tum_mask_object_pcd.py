import argparse
import json
import os
import sys
from datetime import datetime
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

 


'''
python3 3d_construction/fuse_tum_mask_object_pcd.py \
  /home/haochen/code/TSDF/mask_generation/outputs/fr3_office_tv_yolov8x-seg_20260403_144632 \
  --track-id 4

'''

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy")


def default_output_path(config_path):
    output_dir = TSDF_ROOT / "3d_construction" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{Path(config_path).stem}_masked_object.pcd"


def default_time_fuse_dir(config_path):
    output_root = TSDF_ROOT / "3d_construction" / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"time_fuse_{Path(config_path).stem}_{timestamp}"


def default_fuse_obj_dir(mask_source):
    output_root = TSDF_ROOT / "3d_construction" / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_tag = Path(mask_source).name
    return output_root / f"fuse_obj_{source_tag}_{timestamp}"


def infer_config_from_mask_output_dir(mask_output_dir):
    metadata_path = Path(mask_output_dir) / "metadata.json"
    if metadata_path.exists():
        try:
            import json

            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            config_path = metadata.get("config")
            if config_path and Path(config_path).exists():
                return str(Path(config_path))
        except Exception:
            pass

    config_dir = TSDF_ROOT / "configs" / "rgbd" / "tum"
    config_files = sorted(config_dir.glob("*.yaml"))
    folder_name = Path(mask_output_dir).name.lower()
    matches = []
    for config_file in config_files:
        stem = config_file.stem.lower()
        if folder_name.startswith(stem):
            matches.append((len(stem), config_file))
    if matches:
        matches.sort(key=lambda item: item[0], reverse=True)
        return str(matches[0][1])
    return None


def find_pose_list(dataset_path):
    for filename in ("groundtruth.txt", "pose.txt"):
        candidate = os.path.join(dataset_path, filename)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find groundtruth.txt or pose.txt in {dataset_path}"
    )


def load_mask_table(mask_list_path):
    mask_list_path = Path(mask_list_path)
    if not mask_list_path.exists():
        raise FileNotFoundError(f"Mask list file not found: {mask_list_path}")
    if mask_list_path.stat().st_size == 0:
        raise RuntimeError(
            f"Mask list is empty: {mask_list_path}. "
            "If masks were generated as separate instance tracks, use --track-id N or --fuse-all-tracks."
        )

    data = parse_list(mask_list_path)
<<<<<<< HEAD
    data = np.asarray(data)
    if data.size == 0:
        raise RuntimeError(
            f"Mask list is empty: {mask_list_path}. "
            "If the mask generator was run with --separate-instances, use a non-empty "
            "mask_track_*.txt file or run this script with --fuse-all-tracks."
=======
    if data.size == 0:
        raise RuntimeError(
            f"Mask list contains no valid rows: {mask_list_path}. "
            "If masks were generated as separate instance tracks, use --track-id N or --fuse-all-tracks."
>>>>>>> upstream/main
        )
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise RuntimeError(
<<<<<<< HEAD
            f"Mask list must contain at least two columns (timestamp and relative mask path): "
            f"{mask_list_path}"
=======
            f"Mask list must contain at least timestamp and relative path columns: {mask_list_path}"
>>>>>>> upstream/main
        )
    timestamps = data[:, 0].astype(np.float64)
    relpaths = data[:, 1]
    return timestamps, relpaths


def load_multi_mask_table(mask_list_paths):
    all_timestamps = []
    all_relpaths = []
    for mask_list_path in mask_list_paths:
        timestamps, relpaths = load_mask_table(mask_list_path)
        all_timestamps.append(timestamps)
        all_relpaths.append(np.asarray(relpaths, dtype=object))

    if not all_timestamps:
        raise RuntimeError("No mask list files were provided.")

    timestamps = np.concatenate(all_timestamps, axis=0)
    relpaths = np.concatenate(all_relpaths, axis=0)
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
    max_dt=0.03,
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


def load_tum_multi_masked_frames(
    dataset_path,
    mask_dir,
    mask_lists,
    frame_stride=1,
    max_frames=None,
    max_dt=0.03,
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
    tstamp_mask, mask_relpaths = load_multi_mask_table(mask_lists)

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

        match_mask = np.abs(tstamp_mask - rgb_timestamp) < max_dt
        if not np.any(match_mask):
            missing_masks += 1
            continue

        mask_paths = []
        seen_paths = set()
        for relpath in mask_relpaths[match_mask]:
            resolved = resolve_mask_list_path(mask_dir, relpath)
            resolved_str = str(resolved)
            if not Path(resolved).is_file() or resolved_str in seen_paths:
                continue
            seen_paths.add(resolved_str)
            mask_paths.append(resolved_str)

        if not mask_paths:
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
                "mask_paths": mask_paths,
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


def summarize_track_masks(mask_dir, mask_list_path):
    timestamps, mask_relpaths = load_mask_table(mask_list_path)
    areas = []
    coverages = []
    for mask_relpath in mask_relpaths:
        mask_path = resolve_mask_list_path(mask_dir, mask_relpath)
        if mask_path is None or not Path(mask_path).is_file():
            continue
        mask = load_mask(mask_path)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        binary = mask > 0
        area = int(binary.sum())
        if area <= 0:
            continue
        areas.append(area)
        coverages.append(float(area) / float(binary.size))
    return {
        "frames": int(len(areas)),
        "avg_mask_pixels": float(np.mean(areas)) if areas else 0.0,
        "max_mask_pixels": int(max(areas)) if areas else 0,
        "avg_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "max_coverage": float(np.max(coverages)) if coverages else 0.0,
    }


def track_passes_thresholds(track_stats, args):
    return (
        track_stats["frames"] >= args.min_track_frames
        and track_stats["avg_mask_pixels"] >= args.min_track_avg_mask_pixels
        and track_stats["max_mask_pixels"] >= args.min_track_peak_mask_pixels
        and track_stats["avg_coverage"] >= args.min_track_avg_coverage
    )


def auto_select_best_track(mask_output_dir, mask_dir, args):
    track_lists = sorted(Path(mask_output_dir).glob("mask_track_*.txt"))
    if not track_lists:
        return None

    candidates = []
    for track_list in track_lists:
        stats = summarize_track_masks(mask_dir, track_list)
        if args.validate_track and not track_passes_thresholds(stats, args):
            continue
        score = (
            stats["avg_mask_pixels"],
            stats["max_mask_pixels"],
            stats["frames"],
            stats["avg_coverage"],
        )
        candidates.append((score, track_list, stats))

    if not candidates:
        raise RuntimeError(
            "No valid track passed the current thresholds. "
            "Lower the thresholds or choose a track manually with --track-id."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, best_track, best_stats = candidates[0]
    print(
        "Auto-selected track "
        f"{best_track.stem} | frames={best_stats['frames']} "
        f"avg_mask_pixels={best_stats['avg_mask_pixels']:.1f} "
        f"max_mask_pixels={best_stats['max_mask_pixels']} "
        f"avg_coverage={best_stats['avg_coverage']:.6f}"
    )
    return best_track


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


<<<<<<< HEAD
def summarize_rejected_frames(rejected_frames):
    return [
        {
            "timestamp": float(item["timestamp"]),
            "mask_area": int(item["mask_area"]),
            "neighbor_median": float(item["neighbor_median"]),
            "ratio": float(item["ratio"]),
            "reason": item["reason"],
        }
        for item in rejected_frames
    ]


def filter_frames_by_mask_area(mask_areas, timestamps, args):
    if args.mask_area_window <= 0 or len(mask_areas) < 3:
        return np.ones(len(mask_areas), dtype=bool), []

    keep = np.ones(len(mask_areas), dtype=bool)
    rejected = []
    for idx, area in enumerate(mask_areas):
        left = max(0, idx - args.mask_area_window)
        right = min(len(mask_areas), idx + args.mask_area_window + 1)
        neighbor_values = [
            mask_areas[j]
            for j in range(left, right)
            if j != idx and mask_areas[j] > 0
        ]
        if not neighbor_values:
            continue

        neighbor_median = float(np.median(neighbor_values))
        if neighbor_median <= 0:
            continue

        ratio = float(area) / neighbor_median
        reason = None
        if ratio < args.mask_area_ratio_min:
            reason = "below_neighbor_min_ratio"
        elif ratio > args.mask_area_ratio_max:
            reason = "above_neighbor_max_ratio"

        if reason is not None:
            keep[idx] = False
            rejected.append(
                {
                    "timestamp": float(timestamps[idx]),
                    "mask_area": int(area),
                    "neighbor_median": neighbor_median,
                    "ratio": ratio,
                    "reason": reason,
                }
            )
    return keep, rejected


def report_area_filter_stats(total_frames, rejected_frames, label):
    rejected_count = len(rejected_frames)
    kept_count = total_frames - rejected_count
    print(
        f"{label} mask-area filter: kept={kept_count}, rejected={rejected_count}, total={total_frames}"
    )
    if rejected_frames:
        print("Rejected frame timestamps:")
        for item in rejected_frames:
            print(
                f"  ts={item['timestamp']:.6f} area={item['mask_area']} "
                f"neighbor_median={item['neighbor_median']:.1f} ratio={item['ratio']:.3f} "
                f"reason={item['reason']}"
            )


def write_area_filter_report(output_path, frames_loaded, frames_used, empty_mask_frames, rejected_frames):
    report_path = Path(output_path).with_suffix(".area_filter.json")
    payload = {
        "frames_loaded": int(frames_loaded),
        "frames_used": int(frames_used),
        "frames_skipped_empty": int(empty_mask_frames),
        "frames_rejected_area_filter": int(len(rejected_frames)),
        "rejected_frames": summarize_rejected_frames(rejected_frames),
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved area-filter report to {report_path}")


def masked_depth_to_world_points(color, depth, mask, camera, cam_to_world, depth_min, depth_trunc):
=======
def masked_depth_to_world_points(color, depth, mask, camera, cam_to_world, depth_min, depth_trunc, front_quantile=1.0, front_margin=0.0):
>>>>>>> upstream/main
    valid_depth = depth.astype(np.float32) / camera["depth_scale"]
    valid = mask & (valid_depth > depth_min) & (valid_depth < depth_trunc)
    if not np.any(valid):
        return None, None

    v, u = np.nonzero(valid)
    z = valid_depth[v, u]
    if 0.0 < front_quantile < 1.0 and len(z) > 0:
        front_limit = float(np.quantile(z, front_quantile)) + float(front_margin)
        keep = z <= front_limit
        if not np.any(keep):
            return None, None
        v = v[keep]
        u = u[keep]
        z = z[keep]
    x = (u.astype(np.float32) - camera["cx"]) * z / camera["fx"]
    y = (v.astype(np.float32) - camera["cy"]) * z / camera["fy"]

    points_cam = np.stack([x, y, z], axis=1)
    points_world = (
        points_cam @ cam_to_world[:3, :3].T + cam_to_world[:3, 3][None, :]
    )
    colors = color[v, u].astype(np.float32) / 255.0
    return points_world, colors


def keep_largest_dbscan_cluster(point_cloud, eps, min_points):
    if len(point_cloud.points) == 0:
        return point_cloud, {
            "dbscan_clusters": 0,
            "dbscan_kept_points": 0,
            "dbscan_removed_points": 0,
        }

    labels = np.asarray(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    valid = labels >= 0
    if not np.any(valid):
        return point_cloud, {
            "dbscan_clusters": 0,
            "dbscan_kept_points": int(len(point_cloud.points)),
            "dbscan_removed_points": 0,
        }

    cluster_ids, counts = np.unique(labels[valid], return_counts=True)
    keep_cluster = int(cluster_ids[np.argmax(counts)])
    keep_indices = np.where(labels == keep_cluster)[0]
    cleaned = point_cloud.select_by_index(keep_indices.tolist())
    return cleaned, {
        "dbscan_clusters": int(len(cluster_ids)),
        "dbscan_kept_points": int(len(cleaned.points)),
        "dbscan_removed_points": int(len(point_cloud.points) - len(cleaned.points)),
    }


def fuse_from_mask_list(config, dataset_path, args, mask_list_path, output_path):
    if args.validate_track:
        track_stats = summarize_track_masks(args.mask_dir, mask_list_path)
        if track_stats["frames"] < args.min_track_frames:
            raise RuntimeError(
                f"Rejected track {Path(mask_list_path).name}: frames={track_stats['frames']} < min_track_frames={args.min_track_frames}"
            )
        if track_stats["avg_mask_pixels"] < args.min_track_avg_mask_pixels:
            raise RuntimeError(
                f"Rejected track {Path(mask_list_path).name}: avg_mask_pixels={track_stats['avg_mask_pixels']:.1f} < min_track_avg_mask_pixels={args.min_track_avg_mask_pixels}"
            )
        if track_stats["max_mask_pixels"] < args.min_track_peak_mask_pixels:
            raise RuntimeError(
                f"Rejected track {Path(mask_list_path).name}: max_mask_pixels={track_stats['max_mask_pixels']} < min_track_peak_mask_pixels={args.min_track_peak_mask_pixels}"
            )
        if track_stats["avg_coverage"] < args.min_track_avg_coverage:
            raise RuntimeError(
                f"Rejected track {Path(mask_list_path).name}: avg_coverage={track_stats['avg_coverage']:.6f} < min_track_avg_coverage={args.min_track_avg_coverage}"
            )

    frames, missing_masks = load_tum_masked_frames(
        dataset_path=dataset_path,
        mask_dir=args.mask_dir,
        mask_list=mask_list_path,
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

    prepared_frames = []
    mask_areas = []
    timestamps = []
    for frame in frames:
        mask = load_mask(frame["mask_path"])
        mask = preprocess_mask(
            mask,
            camera["width"],
            camera["height"],
            args.mask_threshold,
            args.erode_kernel,
            args.dilate_kernel,
            args.largest_component,
        )
        prepared_frames.append({**frame, "prepared_mask": mask})
        mask_areas.append(int(mask.sum()))
        timestamps.append(frame["timestamp"])

    keep_mask, rejected_frames = filter_frames_by_mask_area(
        mask_areas=np.asarray(mask_areas, dtype=np.int64),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        args=args,
    )
    report_area_filter_stats(len(prepared_frames), rejected_frames, "Single-track")

    for idx, frame in enumerate(prepared_frames, start=1):
        color_bgr = cv2.imread(frame["color_path"], cv2.IMREAD_COLOR)
        if color_bgr is None:
            raise FileNotFoundError(f"Could not read color image: {frame['color_path']}")
        color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        depth = np.array(o3d.io.read_image(frame["depth_path"]))

        color = resize_or_rectify_color(color, camera)
        depth = resize_or_rectify_depth(depth, camera)
        mask = frame["prepared_mask"]

        if not keep_mask[idx - 1]:
            empty_mask_frames += 1
            continue

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
            front_quantile=args.front_depth_quantile,
            front_margin=args.front_depth_margin,
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
    dbscan_summary = {
        "dbscan_clusters": 0,
        "dbscan_kept_points": int(len(point_cloud.points)),
        "dbscan_removed_points": 0,
    }
    if args.dbscan_cleanup:
        point_cloud, dbscan_summary = keep_largest_dbscan_cluster(
            point_cloud,
            eps=args.dbscan_eps,
            min_points=args.dbscan_min_points,
        )
    if args.remove_statistical_outlier:
        point_cloud, _ = point_cloud.remove_statistical_outlier(
            nb_neighbors=args.outlier_nb_neighbors,
            std_ratio=args.outlier_std_ratio,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), point_cloud)
    write_area_filter_report(
        output_path=output_path,
        frames_loaded=len(frames),
        frames_used=used_frames,
        empty_mask_frames=empty_mask_frames,
        rejected_frames=rejected_frames,
    )

    print(f"Saved fused masked object point cloud to {output_path}")
    print(
        f"Frames used: {used_frames}/{len(frames)} | "
        f"points after filtering: {len(point_cloud.points)}"
    )
    return {
        "mask_list": str(mask_list_path) if mask_list_path is not None else None,
        "output": str(output_path),
        "frames_loaded": int(len(frames)),
        "missing_masks": int(missing_masks),
        "frames_used": int(used_frames),
        "frames_skipped_empty": int(empty_mask_frames),
        "frames_rejected_area_filter": int(len(rejected_frames)),
        "rejected_frames": summarize_rejected_frames(rejected_frames),
        "points_after_filtering": int(len(point_cloud.points)),
    }


def fuse_from_mask_lists(config, dataset_path, args, mask_list_paths, output_path):
    frames, missing_masks = load_tum_multi_masked_frames(
        dataset_path=dataset_path,
        mask_dir=args.mask_dir,
        mask_lists=mask_list_paths,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        max_dt=args.max_dt,
    )
    if not frames:
        raise RuntimeError(
            "No synchronized RGB-D-mask frames were found across the track lists."
        )

    camera = build_camera_model(config, args.input_width, args.input_height)
    all_points = []
    all_colors = []
    used_frames = 0
    empty_mask_frames = 0

    print(
        f"Loaded {len(frames)} usable RGB-D-mask frames from {len(mask_list_paths)} track files"
        f" ({missing_masks} RGB frames had no matching mask)."
    )

    prepared_frames = []
    mask_areas = []
    timestamps = []
    for frame in frames:
        merged_mask = np.zeros((camera["height"], camera["width"]), dtype=bool)
        for mask_path in frame["mask_paths"]:
            mask = load_mask(mask_path)
            mask = preprocess_mask(
                mask,
                camera["width"],
                camera["height"],
                args.mask_threshold,
                args.erode_kernel,
                args.dilate_kernel,
                args.largest_component,
            )
            merged_mask |= mask
        prepared_frames.append({**frame, "prepared_mask": merged_mask})
        mask_areas.append(int(merged_mask.sum()))
        timestamps.append(frame["timestamp"])

    keep_mask, rejected_frames = filter_frames_by_mask_area(
        mask_areas=np.asarray(mask_areas, dtype=np.int64),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        args=args,
    )
    report_area_filter_stats(len(prepared_frames), rejected_frames, "Multi-track")

    for idx, frame in enumerate(prepared_frames, start=1):
        color_bgr = cv2.imread(frame["color_path"], cv2.IMREAD_COLOR)
        if color_bgr is None:
            raise FileNotFoundError(f"Could not read color image: {frame['color_path']}")
        color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        depth = np.array(o3d.io.read_image(frame["depth_path"]))

        color = resize_or_rectify_color(color, camera)
        depth = resize_or_rectify_depth(depth, camera)
        merged_mask = frame["prepared_mask"]

        if not keep_mask[idx - 1]:
            empty_mask_frames += 1
            continue

        if int(merged_mask.sum()) < args.min_mask_pixels:
            empty_mask_frames += 1
            continue

        points_world, colors = masked_depth_to_world_points(
            color=color,
            depth=depth,
            mask=merged_mask,
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
            "No 3D object points were reconstructed from the merged track masks."
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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), point_cloud)
    write_area_filter_report(
        output_path=output_path,
        frames_loaded=len(frames),
        frames_used=used_frames,
        empty_mask_frames=empty_mask_frames,
        rejected_frames=rejected_frames,
    )

    print(f"Saved merged multi-track point cloud to {output_path}")
    print(
        f"Frames used: {used_frames}/{len(frames)} | "
        f"points after filtering: {len(point_cloud.points)}"
    )
    return {
        "mask_lists": [str(path) for path in mask_list_paths],
        "output": str(output_path),
        "frames_loaded": int(len(frames)),
        "missing_masks": int(missing_masks),
        "frames_used": int(used_frames),
        "frames_skipped_empty": int(empty_mask_frames),
        "frames_rejected_area_filter": int(len(rejected_frames)),
        "rejected_frames": summarize_rejected_frames(rejected_frames),
        "points_after_filtering": int(len(point_cloud.points)),
        **dbscan_summary,
    }


def infer_track_output_name(mask_list_path):
    mask_list_path = Path(mask_list_path)
    stem = mask_list_path.stem
    return f"{stem}.pcd"


def find_track_list(track_root, track_id):
    track_path = Path(track_root) / f"mask_track_{int(track_id):03d}.txt"
    if not track_path.exists():
        raise FileNotFoundError(
            f"Track file not found: {track_path}. "
            "Check the available mask_track_*.txt files first."
        )
    return track_path


def merge_track_lists(track_root, track_ids, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / (
        "mask_tracks_" + "_".join(f"{int(track_id):03d}" for track_id in track_ids) + ".txt"
    )
    merged_entries = []
    for track_id in track_ids:
        track_path = find_track_list(track_root, track_id)
        with open(track_path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                merged_entries.append((float(parts[0]), stripped))

    if not merged_entries:
        raise RuntimeError(f"No valid mask entries found for track ids {track_ids}.")

    merged_entries.sort(key=lambda item: item[0])
    with open(merged_path, "w", encoding="utf-8") as handle:
        for _, line in merged_entries:
            handle.write(line + "\n")
    return merged_path


def resolve_mask_inputs(args):
    if getattr(args, "mask_output_dir_positional", None) and not args.mask_output_dir:
        args.mask_output_dir = args.mask_output_dir_positional

    mask_output_dir = Path(args.mask_output_dir) if args.mask_output_dir else None

    if mask_output_dir is not None:
        if not mask_output_dir.exists():
            raise FileNotFoundError(f"Mask output folder not found: {mask_output_dir}")

        auto_mask_dir = mask_output_dir / "masks"
        if not auto_mask_dir.exists():
            raise FileNotFoundError(
                f"Could not find masks/ under {mask_output_dir}. "
                "Expected a folder produced by generate_tum_masks_yolo.py."
            )

        args.mask_dir = str(auto_mask_dir)

        if args.fuse_all_tracks:
            track_lists = sorted(mask_output_dir.glob("mask_track_*.txt"))
            if not track_lists:
                raise RuntimeError(
                    f"No mask_track_*.txt files were found under {mask_output_dir}. "
                    "Generate masks with separate instances enabled first."
                )
        else:
            if args.track_ids:
                if len(args.track_ids) == 1:
                    args.mask_list = str(find_track_list(mask_output_dir, args.track_ids[0]))
                else:
                    args.mask_list = str(
                        merge_track_lists(
                            mask_output_dir,
                            args.track_ids,
                            TSDF_ROOT / "3d_construction" / "outputs" / "_tmp_merged_tracks",
                        )
                    )
            elif args.mask_list is None:
                auto_track = auto_select_best_track(mask_output_dir, args.mask_dir, args)
                if auto_track is not None:
                    args.mask_list = str(auto_track)
                else:
                    auto_mask_list = mask_output_dir / "mask.txt"
                    if not auto_mask_list.exists():
                        raise FileNotFoundError(
                            f"Could not find mask.txt under {mask_output_dir}. "
                            "Pass --mask-list explicitly, use --track-id, or use --fuse-all-tracks."
                        )
                    if auto_mask_list.stat().st_size == 0:
                        track_lists = sorted(mask_output_dir.glob("mask_track_*.txt"))
                        if track_lists:
                            raise RuntimeError(
                                f"{auto_mask_list} is empty, but instance track files were found under {mask_output_dir}. "
                                "Use --track-id N to fuse one object or --fuse-all-tracks to fuse every tracked object separately."
                            )
                    args.mask_list = str(auto_mask_list)
    else:
        if args.mask_dir is None:
            raise ValueError("Provide either --mask-output-dir or --mask-dir.")

    if args.config is None and mask_output_dir is not None:
        inferred = infer_config_from_mask_output_dir(mask_output_dir)
        if inferred is None:
            raise ValueError(
                "Could not infer a TUM config from the mask output folder. "
                "Pass --config explicitly."
            )
        args.config = inferred


def fuse_masked_object(args):
    resolve_mask_inputs(args)
    config = load_config(args.config)
    dataset_path = args.dataset or config["Dataset"]["dataset_path"]

    if args.fuse_all_tracks and args.merge_all_tracks:
        raise RuntimeError("Use only one of --fuse-all-tracks or --merge-all-tracks.")
    if args.fuse_all_tracks and args.mask_list:
        raise RuntimeError("--fuse-all-tracks cannot be combined with explicit --mask-list values.")

    if args.fuse_all_tracks:
<<<<<<< HEAD
        if args.mask_list:
            track_root = Path(args.mask_list[0]).parent
=======
        if args.mask_output_dir is not None:
            track_root = Path(args.mask_output_dir)
        elif args.mask_list is not None:
            track_root = Path(args.mask_list).parent
>>>>>>> upstream/main
        else:
            track_root = Path(args.mask_dir).parent

        track_lists = sorted(track_root.glob("mask_track_*.txt"))
        if not track_lists:
            raise RuntimeError(
                f"No mask_track_*.txt files were found under {track_root}. "
                "Run the mask generator with --separate-instances first."
            )

        output_dir = Path(args.output_dir) if args.output_dir else default_time_fuse_dir(
            args.config
        )
        if args.output_dir is None and args.mask_output_dir is not None:
            output_dir = default_fuse_obj_dir(args.mask_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summaries = []

        print(f"Found {len(track_lists)} track files under {track_root}")
        print(f"Saving fused point clouds to {output_dir}")

        for track_list in track_lists:
            print(f"\nFusing {track_list.name}")
            output_path = output_dir / infer_track_output_name(track_list)
            try:
                summary = fuse_from_mask_list(
                    config=config,
                    dataset_path=dataset_path,
                    args=args,
                    mask_list_path=track_list,
                    output_path=output_path,
                )
                summary["track_file"] = track_list.name
                summaries.append(summary)
            except RuntimeError as exc:
                print(f"Skipping {track_list.name}: {exc}")

        if not summaries:
            raise RuntimeError("No track pcd was generated successfully.")

        summary_path = output_dir / "summary.txt"
        with open(summary_path, "w", encoding="utf-8") as handle:
            for item in summaries:
                handle.write(
                    f"{Path(item['output']).name} frames_used={item['frames_used']} "
                    f"points={item['points_after_filtering']} mask_list={Path(item['mask_list']).name}\n"
                )
        print(f"\nSaved {len(summaries)} fused point clouds under {output_dir}")
        print(f"Saved summary to {summary_path}")
        return

<<<<<<< HEAD
    if args.merge_all_tracks:
        if args.mask_list:
            track_root = Path(args.mask_list[0]).parent
        else:
            track_root = Path(args.mask_dir).parent

        track_lists = sorted(track_root.glob("mask_track_*.txt"))
        if not track_lists:
            raise RuntimeError(
                f"No mask_track_*.txt files were found under {track_root}. "
                "Run the mask generator with --separate-instances first."
            )

        if args.output is not None:
            output_path = Path(args.output)
        elif args.output_dir is not None:
            output_path = Path(args.output_dir) / "merged_all_tracks.pcd"
        else:
            output_path = default_output_path(args.config)

        print(f"Found {len(track_lists)} track files under {track_root}")
        print(f"Merging all tracks into one point cloud: {output_path}")
        fuse_from_mask_lists(
            config=config,
            dataset_path=dataset_path,
            args=args,
            mask_list_paths=track_lists,
            output_path=output_path,
        )
        return

    if args.mask_list:
        for raw_mask_list_path in args.mask_list:
            mask_list_path = Path(raw_mask_list_path)
            if mask_list_path.is_file() and mask_list_path.stat().st_size == 0:
                raise RuntimeError(
                    f"Mask list is empty: {mask_list_path}. "
                    "This usually means the masks were generated with --separate-instances, so the "
                    "merged mask.txt was not populated. Use one of the generated mask_track_*.txt files "
                    "for single-object fusion, or pass --fuse-all-tracks."
                )

    output_path = Path(args.output) if args.output else default_output_path(args.config)
    if args.mask_list and len(args.mask_list) > 1:
        fuse_from_mask_lists(
            config=config,
            dataset_path=dataset_path,
            args=args,
            mask_list_paths=[Path(path) for path in args.mask_list],
            output_path=output_path,
        )
    else:
        fuse_from_mask_list(
            config=config,
            dataset_path=dataset_path,
            args=args,
            mask_list_path=args.mask_list[0] if args.mask_list else None,
            output_path=output_path,
        )
=======
    if args.output:
        output_path = Path(args.output)
    elif args.mask_output_dir is not None:
        output_dir = Path(args.output_dir) if args.output_dir else default_fuse_obj_dir(
            args.mask_output_dir
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "fused_object.pcd"
    else:
        output_path = default_output_path(args.config)
    fuse_from_mask_list(
        config=config,
        dataset_path=dataset_path,
        args=args,
        mask_list_path=args.mask_list,
        output_path=output_path,
    )
>>>>>>> upstream/main


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Fuse object point clouds from TUM RGB-D multi-frame masks."
    )
    parser.add_argument(
        "mask_output_dir_positional",
        nargs="?",
        default=None,
        help="Optional positional mask generator output folder containing masks/, mask.txt, and optionally mask_track_*.txt.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a TUM RGB-D config file.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override dataset path from the config file.",
    )
    parser.add_argument(
        "--mask-output-dir",
        default=None,
        help="Mask generator output folder containing masks/, mask.txt, and optionally mask_track_*.txt.",
    )
    parser.add_argument(
        "--mask-dir",
        default=None,
        help="Directory containing masks. Supports mirrored rgb/ paths or flat same-name masks.",
    )
    parser.add_argument(
        "--mask-list",
        action="append",
        default=None,
        help="Optional TUM-style txt file with mask timestamps and relative paths. Repeat this flag to merge multiple track lists into one fused output.",
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
    parser.add_argument(
        "--output-dir",
        default=None,
        help="When --fuse-all-tracks is set, save all fused track pcd files under this directory. If --mask-output-dir is used, the default is a new outputs/fuse_obj_* folder.",
    )
    parser.add_argument(
        "--fuse-all-tracks",
        action="store_true",
        help="Fuse every mask_track_*.txt under the mask output folder into separate pcd files.",
    )
    parser.add_argument(
<<<<<<< HEAD
        "--merge-all-tracks",
        action="store_true",
        help="Merge all mask_track_*.txt masks per frame and fuse them into one combined pcd.",
=======
        "--track-id",
        type=int,
        dest="track_ids",
        action="append",
        default=[],
        help="Fuse one or more tracked instances. Repeat the flag, for example --track-id 0 --track-id 1.",
>>>>>>> upstream/main
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-dt", type=float, default=0.08)
    parser.add_argument("--input-width", type=int, default=None)
    parser.add_argument("--input-height", type=int, default=None)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
<<<<<<< HEAD
    parser.add_argument("--min-mask-pixels", type=int, default=64)
    parser.add_argument(
        "--mask-area-window",
        type=int,
        default=2,
        help="Use +/- this many neighboring frames to estimate the expected mask area. Set 0 to disable area-based filtering.",
    )
    parser.add_argument(
        "--mask-area-ratio-min",
        type=float,
        default=0.5,
        help="Reject a frame when its mask area is smaller than this fraction of the neighboring-frame median area.",
    )
    parser.add_argument(
        "--mask-area-ratio-max",
        type=float,
        default=1.8,
        help="Reject a frame when its mask area is larger than this multiple of the neighboring-frame median area.",
    )
=======
    parser.add_argument("--min-mask-pixels", type=int, default=1500)
>>>>>>> upstream/main
    parser.add_argument("--erode-kernel", type=int, default=0)
    parser.add_argument("--dilate-kernel", type=int, default=0)
    parser.add_argument(
        "--largest-component",
        dest="largest_component",
        action="store_true",
        help="Keep only the largest connected mask component. Enabled by default.",
    )
    parser.add_argument(
        "--keep-all-components",
        dest="largest_component",
        action="store_false",
        help="Keep all mask components instead of only the largest one.",
    )
    parser.add_argument(
        "--front-depth-quantile",
        type=float,
        default=0.7,
        help="Keep only the front part of masked depth values up to this quantile. Use values like 0.7-0.9 to suppress deeper background leakage.",
    )
    parser.add_argument(
        "--front-depth-margin",
        type=float,
        default=0.03,
        help="Extra depth margin in meters added after front-depth quantile filtering.",
    )
    parser.add_argument("--depth-min", type=float, default=0.05)
    parser.add_argument("--depth-trunc", type=float, default=4.0)
    parser.add_argument(
        "--validate-track",
        dest="validate_track",
        action="store_true",
        help="Reject weak tracks using frame count and mask coverage thresholds before 3D fusion. Enabled by default.",
    )
    parser.add_argument(
        "--no-validate-track",
        dest="validate_track",
        action="store_false",
        help="Disable track validation before 3D fusion.",
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=10,
        help="Minimum number of masked frames required for a track to be fused when --validate-track is enabled.",
    )
    parser.add_argument(
        "--min-track-avg-mask-pixels",
        type=float,
        default=2000.0,
        help="Minimum average mask pixel count required for a valid track when --validate-track is enabled.",
    )
    parser.add_argument(
        "--min-track-peak-mask-pixels",
        type=int,
        default=6000,
        help="Minimum peak mask pixel count required for a valid track when --validate-track is enabled.",
    )
    parser.add_argument(
        "--min-track-avg-coverage",
        type=float,
        default=0.003,
        help="Minimum average image coverage ratio required for a valid track when --validate-track is enabled.",
    )
    parser.add_argument("--voxel-downsample", type=float, default=0.005)
    parser.add_argument(
        "--dbscan-cleanup",
        dest="dbscan_cleanup",
        action="store_true",
        help="Keep only the largest DBSCAN cluster before saving. Enabled by default.",
    )
    parser.add_argument(
        "--no-dbscan-cleanup",
        dest="dbscan_cleanup",
        action="store_false",
        help="Disable DBSCAN cleanup before saving.",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.03,
        help="DBSCAN epsilon in meters used for fused point cloud cleanup.",
    )
    parser.add_argument(
        "--dbscan-min-points",
        type=int,
        default=80,
        help="Minimum number of neighbors for DBSCAN cleanup.",
    )
    parser.add_argument(
        "--remove-statistical-outlier",
        dest="remove_statistical_outlier",
        action="store_true",
        help="Apply statistical outlier removal after fusion. Enabled by default.",
    )
    parser.add_argument(
        "--no-remove-statistical-outlier",
        dest="remove_statistical_outlier",
        action="store_false",
        help="Disable statistical outlier removal after fusion.",
    )
    parser.add_argument("--outlier-nb-neighbors", type=int, default=20)
    parser.add_argument("--outlier-std-ratio", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.set_defaults(
        validate_track=True,
        largest_component=True,
        dbscan_cleanup=True,
        remove_statistical_outlier=True,
    )
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    fuse_masked_object(parser.parse_args())
