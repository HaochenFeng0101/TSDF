import argparse
import io
import json
import math
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import cv2
import yaml
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

'''
python 3d_construction/compare_tsdf_openmvs_no_gt.py \
  --tsdf-pcd 3d_construction/outputs/fr3_office.pcd \
  --openmvs-workspace openmvs/workspaces/fr3_office_openmvs

'''

TSDF_ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path_str):
    path = Path(path_str)
    if not path.is_absolute():
        path = (TSDF_ROOT / path).resolve()
    return path


def default_output_dir(tsdf_path, workspace_path):
    del tsdf_path
    del workspace_path
    return TSDF_ROOT / "3d_construction" / "eval"


def reset_output_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_workspace_info(workspace_dir):
    info_path = workspace_dir / "workspace_info.txt"
    info = {}
    if not info_path.exists():
        return info
    for line in info_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        info[key.strip()] = value.strip()
    return info


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


def parse_list(filepath, skiprows=0):
    lines = Path(filepath).read_text(encoding="utf-8").splitlines()
    clean_lines = []
    for line in lines[skiprows:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        clean_lines.append(stripped)
    return np.loadtxt(
        io.StringIO("\n".join(clean_lines)),
        delimiter=" ",
        dtype=np.str_,
    )


def associate_frames(tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
    associations = []
    for i, tstamp in enumerate(tstamp_image):
        depth_idx = np.argmin(np.abs(tstamp_depth - tstamp))
        pose_idx = np.argmin(np.abs(tstamp_pose - tstamp))
        if (
            np.abs(tstamp_depth[depth_idx] - tstamp) < max_dt
            and np.abs(tstamp_pose[pose_idx] - tstamp) < max_dt
        ):
            associations.append((i, depth_idx, pose_idx))
    return associations


def pose_to_world_to_cam(tx, ty, tz, qx, qy, qz, qw):
    rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    cam_to_world = np.eye(4, dtype=np.float64)
    cam_to_world[:3, :3] = rotation
    cam_to_world[:3, 3] = [tx, ty, tz]
    return np.linalg.inv(cam_to_world)


def load_tum_eval_frames(dataset_path, frame_step=25, max_frames=30, max_dt=0.08):
    dataset_path = Path(dataset_path)
    pose_list = dataset_path / "groundtruth.txt"
    if not pose_list.exists():
        pose_list = dataset_path / "pose.txt"
    if not pose_list.exists():
        raise FileNotFoundError(
            f"Could not find groundtruth.txt or pose.txt in {dataset_path}"
        )

    image_list = dataset_path / "rgb.txt"
    depth_list = dataset_path / "depth.txt"
    if not image_list.exists() or not depth_list.exists():
        raise FileNotFoundError(f"Missing rgb.txt or depth.txt in {dataset_path}")

    image_data = parse_list(image_list)
    depth_data = parse_list(depth_list)
    pose_data = parse_list(pose_list, skiprows=1)

    tstamp_image = image_data[:, 0].astype(np.float64)
    tstamp_depth = depth_data[:, 0].astype(np.float64)
    tstamp_pose = pose_data[:, 0].astype(np.float64)
    pose_vecs = pose_data[:, :].astype(np.float64)

    associations = associate_frames(tstamp_image, tstamp_depth, tstamp_pose, max_dt)
    if frame_step > 1:
        associations = associations[::frame_step]
    if max_frames is not None and max_frames > 0:
        associations = associations[:max_frames]

    frames = []
    for image_idx, depth_idx, pose_idx in associations:
        tx, ty, tz = pose_vecs[pose_idx][1:4]
        qx, qy, qz, qw = pose_vecs[pose_idx][4:8]
        frames.append(
            {
                "depth_path": str(dataset_path / depth_data[depth_idx, 1]),
                "extrinsic": pose_to_world_to_cam(tx, ty, tz, qx, qy, qz, qw),
                "timestamp": float(tstamp_image[image_idx]),
            }
        )
    return frames


def build_camera_model(config):
    calibration = config["Dataset"]["Calibration"]
    width = int(calibration["width"])
    height = int(calibration["height"])
    return {
        "width": width,
        "height": height,
        "fx": float(calibration["fx"]),
        "fy": float(calibration["fy"]),
        "cx": float(calibration["cx"]),
        "cy": float(calibration["cy"]),
        "depth_scale": float(calibration["depth_scale"]),
        "distorted": bool(calibration.get("distorted", False)),
        "dist_coeffs": np.array(
            [
                float(calibration.get("k1", 0.0)),
                float(calibration.get("k2", 0.0)),
                float(calibration.get("p1", 0.0)),
                float(calibration.get("p2", 0.0)),
                float(calibration.get("k3", 0.0)),
            ],
            dtype=np.float32,
        ),
        "K": np.array(
            [
                [float(calibration["fx"]), 0.0, float(calibration["cx"])],
                [0.0, float(calibration["fy"]), float(calibration["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    }


def load_depth_meters(depth_path, camera):
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth image: {depth_path}")
    if depth.ndim != 2:
        raise ValueError(f"Depth image must be single channel: {depth_path}")
    if depth.shape[1] != camera["width"] or depth.shape[0] != camera["height"]:
        depth = cv2.resize(
            depth,
            (camera["width"], camera["height"]),
            interpolation=cv2.INTER_NEAREST,
        )
    return depth.astype(np.float32) / camera["depth_scale"]


def render_depth_from_cloud(points, extrinsic, camera):
    points_h = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float64)], axis=1
    )
    points_cam = (extrinsic @ points_h.T).T[:, :3]
    valid = points_cam[:, 2] > 1e-4
    points_cam = points_cam[valid]
    if len(points_cam) == 0:
        return np.zeros((camera["height"], camera["width"]), dtype=np.float32)

    u = np.rint(camera["fx"] * (points_cam[:, 0] / points_cam[:, 2]) + camera["cx"]).astype(np.int32)
    v = np.rint(camera["fy"] * (points_cam[:, 1] / points_cam[:, 2]) + camera["cy"]).astype(np.int32)
    inside = (
        (u >= 0)
        & (u < camera["width"])
        & (v >= 0)
        & (v < camera["height"])
    )
    u = u[inside]
    v = v[inside]
    z = points_cam[:, 2][inside]
    if len(z) == 0:
        return np.zeros((camera["height"], camera["width"]), dtype=np.float32)

    linear = v * camera["width"] + u
    order = np.lexsort((z, linear))
    linear_sorted = linear[order]
    z_sorted = z[order]
    unique_linear, first_idx = np.unique(linear_sorted, return_index=True)

    rendered = np.zeros((camera["height"], camera["width"]), dtype=np.float32)
    rendered.flat[unique_linear] = z_sorted[first_idx].astype(np.float32)
    return rendered


def compute_openmvs_slam_metrics(cloud, workspace_info, thresholds, frame_step, max_frames):
    config_path = workspace_info.get("config")
    dataset_path = workspace_info.get("dataset")
    if not config_path or not dataset_path:
        return None

    config = load_config(config_path)
    camera = build_camera_model(config)
    frames = load_tum_eval_frames(dataset_path, frame_step=frame_step, max_frames=max_frames)
    points = get_points(cloud)

    per_frame = []
    all_errors = []
    total_sensor_valid = 0
    total_render_valid = 0
    total_overlap = 0

    for frame in frames:
        depth = load_depth_meters(frame["depth_path"], camera)
        rendered = render_depth_from_cloud(points, frame["extrinsic"], camera)
        sensor_valid = depth > 1e-4
        render_valid = rendered > 1e-4
        overlap = sensor_valid & render_valid

        sensor_valid_count = int(sensor_valid.sum())
        render_valid_count = int(render_valid.sum())
        overlap_count = int(overlap.sum())

        total_sensor_valid += sensor_valid_count
        total_render_valid += render_valid_count
        total_overlap += overlap_count

        record = {
            "timestamp": frame["timestamp"],
            "sensor_valid_pixels": sensor_valid_count,
            "render_valid_pixels": render_valid_count,
            "overlap_pixels": overlap_count,
            "sensor_overlap_ratio": float(overlap_count / sensor_valid_count)
            if sensor_valid_count > 0
            else 0.0,
            "render_overlap_ratio": float(overlap_count / render_valid_count)
            if render_valid_count > 0
            else 0.0,
        }

        if overlap_count > 0:
            errors = np.abs(rendered[overlap] - depth[overlap]).astype(np.float64)
            all_errors.append(errors)
            record["mae_m"] = float(np.mean(errors))
            record["median_m"] = float(np.median(errors))
            record["rmse_m"] = float(np.sqrt(np.mean(np.square(errors))))
            for threshold in thresholds:
                record[f"within_{threshold:.3f}m"] = float(np.mean(errors <= threshold))
        else:
            record["mae_m"] = None
            record["median_m"] = None
            record["rmse_m"] = None
            for threshold in thresholds:
                record[f"within_{threshold:.3f}m"] = None
        per_frame.append(record)

    summary = {
        "num_eval_frames": len(frames),
        "total_sensor_valid_pixels": total_sensor_valid,
        "total_render_valid_pixels": total_render_valid,
        "total_overlap_pixels": total_overlap,
        "sensor_overlap_ratio": float(total_overlap / total_sensor_valid)
        if total_sensor_valid > 0
        else 0.0,
        "render_overlap_ratio": float(total_overlap / total_render_valid)
        if total_render_valid > 0
        else 0.0,
        "thresholds_m": list(thresholds),
        "per_frame": per_frame,
    }

    if all_errors:
        errors = np.concatenate(all_errors, axis=0)
        summary["mae_m"] = float(np.mean(errors))
        summary["median_m"] = float(np.median(errors))
        summary["rmse_m"] = float(np.sqrt(np.mean(np.square(errors))))
        summary["p90_m"] = float(np.percentile(errors, 90))
        for threshold in thresholds:
            summary[f"within_{threshold:.3f}m"] = float(np.mean(errors <= threshold))
    else:
        summary["mae_m"] = None
        summary["median_m"] = None
        summary["rmse_m"] = None
        summary["p90_m"] = None
        for threshold in thresholds:
            summary[f"within_{threshold:.3f}m"] = None

    return summary


def load_point_cloud(path):
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"Empty point cloud: {path}")
    cloud = cloud.remove_non_finite_points()
    return cloud


def get_points(cloud):
    return np.asarray(cloud.points, dtype=np.float64)


def get_colors(cloud):
    colors = np.asarray(cloud.colors, dtype=np.float64)
    if colors.size == 0:
        return None
    return colors


def prepare_cloud(cloud, voxel_size, normal_radius, max_points):
    prepared = cloud.voxel_down_sample(voxel_size=voxel_size) if voxel_size > 0 else cloud
    points = get_points(prepared)
    if len(points) == 0:
        raise ValueError("Point cloud became empty after downsampling")
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(42)
        keep = rng.choice(len(points), size=max_points, replace=False)
        prepared = prepared.select_by_index(keep.tolist())
    prepared.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=30,
        )
    )
    prepared.normalize_normals()
    return prepared


def write_standard_ascii_ply(cloud, path):
    success = o3d.io.write_point_cloud(
        str(path),
        cloud,
        write_ascii=True,
        compressed=False,
        print_progress=False,
    )
    if not success:
        raise IOError(f"Failed to write point cloud: {path}")


def bbox_metrics(points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extents = maxs - mins
    bbox_volume = float(np.prod(np.maximum(extents, 1e-8)))
    centroid = points.mean(axis=0)
    return {
        "min_xyz": mins.tolist(),
        "max_xyz": maxs.tolist(),
        "extent_xyz": extents.tolist(),
        "bbox_volume_m3": bbox_volume,
        "centroid_xyz": centroid.tolist(),
    }


def k_distance_stats(points, k=2):
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k)
    neighbor_dist = distances[:, -1]
    return {
        "mean_nn_m": float(np.mean(neighbor_dist)),
        "median_nn_m": float(np.median(neighbor_dist)),
        "p90_nn_m": float(np.percentile(neighbor_dist, 90)),
    }


def voxel_occupancy(points, voxel_size):
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    coords = np.floor(points / voxel_size).astype(np.int64)
    return {tuple(v) for v in coords}


def one_way_metrics(src_points, dst_points, src_normals, dst_normals):
    tree = cKDTree(dst_points)
    distances, indices = tree.query(src_points, k=1)
    matched_normals = dst_normals[indices]
    cosine = np.clip(np.sum(src_normals * matched_normals, axis=1), -1.0, 1.0)
    cosine_abs = np.abs(cosine)
    return {
        "distances": distances,
        "cosine_abs": cosine_abs,
    }


def summarize_distances(distances, thresholds):
    summary = {
        "mean_m": float(np.mean(distances)),
        "median_m": float(np.median(distances)),
        "rmse_m": float(np.sqrt(np.mean(np.square(distances)))),
        "p90_m": float(np.percentile(distances, 90)),
        "p95_m": float(np.percentile(distances, 95)),
    }
    for threshold in thresholds:
        key = f"within_{threshold:.3f}m"
        summary[key] = float(np.mean(distances <= threshold))
    return summary


def summarize_normals(cosine_abs):
    angles_deg = np.degrees(np.arccos(np.clip(cosine_abs, -1.0, 1.0)))
    return {
        "mean_abs_cosine": float(np.mean(cosine_abs)),
        "median_angle_deg": float(np.median(angles_deg)),
        "p90_angle_deg": float(np.percentile(angles_deg, 90)),
    }


def compute_mutual_metrics(tsdf_cloud, openmvs_cloud, voxel_size, thresholds):
    tsdf_points = get_points(tsdf_cloud)
    openmvs_points = get_points(openmvs_cloud)
    tsdf_normals = np.asarray(tsdf_cloud.normals, dtype=np.float64)
    openmvs_normals = np.asarray(openmvs_cloud.normals, dtype=np.float64)

    tsdf_to_openmvs = one_way_metrics(
        tsdf_points, openmvs_points, tsdf_normals, openmvs_normals
    )
    openmvs_to_tsdf = one_way_metrics(
        openmvs_points, tsdf_points, openmvs_normals, tsdf_normals
    )

    tsdf_voxels = voxel_occupancy(tsdf_points, voxel_size)
    openmvs_voxels = voxel_occupancy(openmvs_points, voxel_size)
    intersection = len(tsdf_voxels & openmvs_voxels)
    union = len(tsdf_voxels | openmvs_voxels)

    symmetric_chamfer = float(
        np.mean(tsdf_to_openmvs["distances"]) + np.mean(openmvs_to_tsdf["distances"])
    )
    symmetric_rmse = float(
        math.sqrt(
            0.5
            * (
                np.mean(np.square(tsdf_to_openmvs["distances"]))
                + np.mean(np.square(openmvs_to_tsdf["distances"]))
            )
        )
    )

    metrics = {
        "voxel_size_m": voxel_size,
        "distance_thresholds_m": list(thresholds),
        "tsdf_to_openmvs": summarize_distances(tsdf_to_openmvs["distances"], thresholds),
        "openmvs_to_tsdf": summarize_distances(
            openmvs_to_tsdf["distances"], thresholds
        ),
        "tsdf_to_openmvs_normal": summarize_normals(tsdf_to_openmvs["cosine_abs"]),
        "openmvs_to_tsdf_normal": summarize_normals(openmvs_to_tsdf["cosine_abs"]),
        "symmetric_chamfer_l1_m": symmetric_chamfer,
        "symmetric_rmse_m": symmetric_rmse,
        "occupancy_intersection": intersection,
        "occupancy_union": union,
        "occupancy_iou": float(intersection / union) if union > 0 else 0.0,
        "tsdf_voxel_recall": float(intersection / len(tsdf_voxels))
        if tsdf_voxels
        else 0.0,
        "openmvs_voxel_recall": float(intersection / len(openmvs_voxels))
        if openmvs_voxels
        else 0.0,
        "_arrays": {
            "tsdf_to_openmvs_distances": tsdf_to_openmvs["distances"],
            "openmvs_to_tsdf_distances": openmvs_to_tsdf["distances"],
            "tsdf_to_openmvs_cosine_abs": tsdf_to_openmvs["cosine_abs"],
            "openmvs_to_tsdf_cosine_abs": openmvs_to_tsdf["cosine_abs"],
        },
    }
    return metrics


def random_sample(points, max_points, seed):
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(points), size=max_points, replace=False)
    return points[keep]


def plot_overview(tsdf_points, openmvs_points, output_path):
    tsdf_xy = random_sample(tsdf_points[:, :2], 25000, 1)
    openmvs_xy = random_sample(openmvs_points[:, :2], 25000, 2)
    tsdf_xz = random_sample(tsdf_points[:, [0, 2]], 25000, 3)
    openmvs_xz = random_sample(openmvs_points[:, [0, 2]], 25000, 4)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=180)
    axes[0].scatter(openmvs_xy[:, 0], openmvs_xy[:, 1], s=1, alpha=0.25, label="OpenMVS")
    axes[0].scatter(tsdf_xy[:, 0], tsdf_xy[:, 1], s=1, alpha=0.25, label="TSDF")
    axes[0].set_title("Top View (X-Y)")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(
        openmvs_xz[:, 0], openmvs_xz[:, 1], s=1, alpha=0.25, label="OpenMVS"
    )
    axes[1].scatter(tsdf_xz[:, 0], tsdf_xz[:, 1], s=1, alpha=0.25, label="TSDF")
    axes[1].set_title("Side View (X-Z)")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Z (m)")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_consistency(metrics, thresholds, output_path):
    arrays = metrics["_arrays"]
    tsdf_to_openmvs = arrays["tsdf_to_openmvs_distances"]
    openmvs_to_tsdf = arrays["openmvs_to_tsdf_distances"]
    tsdf_angles = np.degrees(np.arccos(np.clip(arrays["tsdf_to_openmvs_cosine_abs"], -1.0, 1.0)))
    openmvs_angles = np.degrees(
        np.arccos(np.clip(arrays["openmvs_to_tsdf_cosine_abs"], -1.0, 1.0))
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)

    bins = np.linspace(
        0.0,
        max(
            np.percentile(tsdf_to_openmvs, 99),
            np.percentile(openmvs_to_tsdf, 99),
            thresholds[-1] * 1.5,
        ),
        60,
    )
    axes[0, 0].hist(tsdf_to_openmvs, bins=bins, alpha=0.6, label="TSDF -> OpenMVS")
    axes[0, 0].hist(openmvs_to_tsdf, bins=bins, alpha=0.6, label="OpenMVS -> TSDF")
    axes[0, 0].set_title("Nearest-Neighbor Distance")
    axes[0, 0].set_xlabel("Distance (m)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend(loc="best")
    axes[0, 0].grid(alpha=0.25)

    coverage_tsdf = [
        metrics["tsdf_to_openmvs"][f"within_{threshold:.3f}m"] for threshold in thresholds
    ]
    coverage_openmvs = [
        metrics["openmvs_to_tsdf"][f"within_{threshold:.3f}m"] for threshold in thresholds
    ]
    axes[0, 1].plot(thresholds, coverage_tsdf, marker="o", label="TSDF -> OpenMVS")
    axes[0, 1].plot(
        thresholds, coverage_openmvs, marker="o", label="OpenMVS -> TSDF"
    )
    axes[0, 1].set_title("Coverage Within Distance Threshold")
    axes[0, 1].set_xlabel("Threshold (m)")
    axes[0, 1].set_ylabel("Fraction")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].legend(loc="best")
    axes[0, 1].grid(alpha=0.25)

    angle_bins = np.linspace(0.0, 90.0, 45)
    axes[1, 0].hist(tsdf_angles, bins=angle_bins, alpha=0.6, label="TSDF -> OpenMVS")
    axes[1, 0].hist(openmvs_angles, bins=angle_bins, alpha=0.6, label="OpenMVS -> TSDF")
    axes[1, 0].set_title("Normal-Angle Error")
    axes[1, 0].set_xlabel("Angle (deg)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].legend(loc="best")
    axes[1, 0].grid(alpha=0.25)

    bar_labels = [
        "Chamfer-L1",
        "Sym. RMSE",
        "Voxel IoU",
        "TSDF voxel recall",
        "OpenMVS voxel recall",
    ]
    bar_values = [
        metrics["symmetric_chamfer_l1_m"],
        metrics["symmetric_rmse_m"],
        metrics["occupancy_iou"],
        metrics["tsdf_voxel_recall"],
        metrics["openmvs_voxel_recall"],
    ]
    axes[1, 1].bar(bar_labels, bar_values, color=["C0", "C1", "C2", "C3", "C4"])
    axes[1, 1].set_title("Summary Metrics")
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_openmvs_slam_metrics(slam_metrics, output_path):
    per_frame = slam_metrics.get("per_frame", [])
    if not per_frame:
        return

    frame_ids = np.arange(len(per_frame))
    sensor_overlap = [item["sensor_overlap_ratio"] for item in per_frame]
    render_overlap = [item["render_overlap_ratio"] for item in per_frame]
    mae = [np.nan if item["mae_m"] is None else item["mae_m"] for item in per_frame]
    rmse = [np.nan if item["rmse_m"] is None else item["rmse_m"] for item in per_frame]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=180)

    axes[0, 0].plot(frame_ids, sensor_overlap, marker="o", label="sensor overlap ratio")
    axes[0, 0].plot(frame_ids, render_overlap, marker="o", label="render overlap ratio")
    axes[0, 0].set_title("OpenMVS Frame Overlap")
    axes[0, 0].set_xlabel("Eval frame index")
    axes[0, 0].set_ylabel("Ratio")
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].legend(loc="best")
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(frame_ids, mae, marker="o", label="MAE")
    axes[0, 1].plot(frame_ids, rmse, marker="o", label="RMSE")
    axes[0, 1].set_title("OpenMVS Depth Error")
    axes[0, 1].set_xlabel("Eval frame index")
    axes[0, 1].set_ylabel("Error (m)")
    axes[0, 1].legend(loc="best")
    axes[0, 1].grid(alpha=0.25)

    labels = ["MAE", "Median", "RMSE", "P90"]
    values = [
        slam_metrics.get("mae_m"),
        slam_metrics.get("median_m"),
        slam_metrics.get("rmse_m"),
        slam_metrics.get("p90_m"),
    ]
    values = [0.0 if value is None else value for value in values]
    axes[1, 0].bar(labels, values, color=["C0", "C1", "C2", "C3"])
    axes[1, 0].set_title("OpenMVS SLAM Summary Error")
    axes[1, 0].set_ylabel("Meters")
    axes[1, 0].grid(axis="y", alpha=0.25)

    threshold_keys = [
        key
        for key in slam_metrics.keys()
        if key.startswith("within_") and key.endswith("m")
    ]
    threshold_keys = sorted(threshold_keys, key=lambda key: float(key.split("_")[1][:-1]))
    threshold_vals = [slam_metrics[key] for key in threshold_keys]
    threshold_vals = [0.0 if value is None else value for value in threshold_vals]
    axes[1, 1].bar(threshold_keys, threshold_vals, color="C4")
    axes[1, 1].set_title("OpenMVS Inlier Ratio")
    axes[1, 1].set_ylabel("Fraction")
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def strip_array_payload(metrics):
    clean = {}
    for key, value in metrics.items():
        if key == "_arrays":
            continue
        if isinstance(value, dict):
            clean[key] = strip_array_payload(value)
        else:
            clean[key] = value
    return clean


def write_summary(report, output_path):
    workspace_info = report["workspace_info"]
    lines = [
        "No-GT Reconstruction Comparison",
        "",
        f"TSDF input: {report['inputs']['tsdf_pcd']}",
        f"OpenMVS workspace: {report['inputs']['openmvs_workspace']}",
        f"OpenMVS dense cloud: {report['inputs']['openmvs_dense_cloud']}",
    ]
    if workspace_info:
        lines.append("")
        lines.append("Workspace info:")
        for key in sorted(workspace_info):
            lines.append(f"  {key}: {workspace_info[key]}")

    metrics = report["mutual_consistency"]
    lines.extend(
        [
            "",
            "Key metrics:",
            f"  symmetric_chamfer_l1_m: {metrics['symmetric_chamfer_l1_m']:.6f}",
            f"  symmetric_rmse_m: {metrics['symmetric_rmse_m']:.6f}",
            f"  occupancy_iou: {metrics['occupancy_iou']:.4f}",
            f"  tsdf_voxel_recall: {metrics['tsdf_voxel_recall']:.4f}",
            f"  openmvs_voxel_recall: {metrics['openmvs_voxel_recall']:.4f}",
            "",
            "Directional distances:",
            f"  TSDF -> OpenMVS mean_m: {metrics['tsdf_to_openmvs']['mean_m']:.6f}",
            f"  OpenMVS -> TSDF mean_m: {metrics['openmvs_to_tsdf']['mean_m']:.6f}",
            f"  TSDF -> OpenMVS median_m: {metrics['tsdf_to_openmvs']['median_m']:.6f}",
            f"  OpenMVS -> TSDF median_m: {metrics['openmvs_to_tsdf']['median_m']:.6f}",
            "",
            "How to read:",
            "  Lower Chamfer / RMSE is better.",
            "  Higher occupancy IoU / voxel recall is better.",
            "  Lower normal-angle error means local surface direction agrees better.",
            "  These are relative consistency metrics, not absolute accuracy, because no ground truth is used.",
        ]
    )
    slam_metrics = report.get("openmvs_slam_view_consistency")
    if slam_metrics is not None:
        lines.extend(
            [
                "",
                "OpenMVS SLAM-view metrics:",
                f"  num_eval_frames: {slam_metrics['num_eval_frames']}",
                f"  sensor_overlap_ratio: {slam_metrics['sensor_overlap_ratio']:.4f}",
                f"  render_overlap_ratio: {slam_metrics['render_overlap_ratio']:.4f}",
                f"  mae_m: {0.0 if slam_metrics['mae_m'] is None else slam_metrics['mae_m']:.6f}",
                f"  median_m: {0.0 if slam_metrics['median_m'] is None else slam_metrics['median_m']:.6f}",
                f"  rmse_m: {0.0 if slam_metrics['rmse_m'] is None else slam_metrics['rmse_m']:.6f}",
                "",
                "How to read OpenMVS SLAM-view metrics:",
                "  sensor_overlap_ratio: how much of valid RGB-D depth is explained by the OpenMVS cloud.",
                "  render_overlap_ratio: how much of the rendered OpenMVS depth lands on valid sensor depth.",
                "  MAE / RMSE: reprojection depth error against RGB-D frames using known poses.",
                "  within_xxm: fraction of overlap pixels whose depth error stays under the threshold.",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(args, tsdf_raw, openmvs_raw, tsdf_prepared, openmvs_prepared, metrics, workspace_info):
    tsdf_raw_points = get_points(tsdf_raw)
    openmvs_raw_points = get_points(openmvs_raw)
    tsdf_points = get_points(tsdf_prepared)
    openmvs_points = get_points(openmvs_prepared)

    report = {
        "inputs": {
            "tsdf_pcd": str(args.tsdf_pcd),
            "openmvs_workspace": str(args.openmvs_workspace),
            "openmvs_dense_cloud": str(args.openmvs_dense_cloud),
            "voxel_size_m": args.voxel_size,
            "normal_radius_m": args.normal_radius,
            "max_points": args.max_points,
        },
        "workspace_info": workspace_info,
        "tsdf": {
            "raw_num_points": int(len(tsdf_raw_points)),
            "prepared_num_points": int(len(tsdf_points)),
            "has_color": get_colors(tsdf_raw) is not None,
            **bbox_metrics(tsdf_points),
            **k_distance_stats(tsdf_points),
        },
        "openmvs": {
            "raw_num_points": int(len(openmvs_raw_points)),
            "prepared_num_points": int(len(openmvs_points)),
            "has_color": get_colors(openmvs_raw) is not None,
            **bbox_metrics(openmvs_points),
            **k_distance_stats(openmvs_points),
        },
        "mutual_consistency": strip_array_payload(metrics),
    }
    return report


def parse_thresholds(values):
    thresholds = sorted({round(float(v), 6) for v in values if float(v) > 0})
    if not thresholds:
        raise ValueError("At least one positive threshold is required")
    return thresholds


def main():
    parser = argparse.ArgumentParser(
        description="Compare TSDF and OpenMVS reconstructions without ground truth."
    )
    parser.add_argument(
        "--tsdf-pcd",
        type=resolve_path,
        required=True,
        help="Path to the TSDF output point cloud (.pcd/.ply).",
    )
    parser.add_argument(
        "--openmvs-workspace",
        type=resolve_path,
        required=True,
        help="Path to the OpenMVS workspace directory.",
    )
    parser.add_argument(
        "--openmvs-dense-cloud",
        type=resolve_path,
        default=None,
        help="Optional explicit dense cloud path. Defaults to <workspace>/scene_dense.ply.",
    )
    parser.add_argument(
        "--output-dir",
        type=resolve_path,
        default=None,
        help="Directory to save JSON/TXT reports and figures.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.03,
        help="Voxel size used for downsampling and voxel-overlap metrics, in meters.",
    )
    parser.add_argument(
        "--normal-radius",
        type=float,
        default=0.08,
        help="Radius used for normal estimation, in meters.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=120000,
        help="Maximum number of downsampled points kept per cloud for evaluation.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.10],
        help="Distance thresholds, in meters, used for directional coverage statistics.",
    )
    parser.add_argument(
        "--slam-eval-frame-step",
        type=int,
        default=25,
        help="Evaluate one RGB-D frame every N synchronized frames for OpenMVS SLAM-view metrics.",
    )
    parser.add_argument(
        "--slam-eval-max-frames",
        type=int,
        default=30,
        help="Maximum number of RGB-D frames used for OpenMVS SLAM-view metrics.",
    )
    args = parser.parse_args()

    workspace_dir = args.openmvs_workspace
    if not workspace_dir.is_dir():
        raise FileNotFoundError(f"OpenMVS workspace not found: {workspace_dir}")

    args.openmvs_dense_cloud = args.openmvs_dense_cloud or (workspace_dir / "scene_dense.ply")
    if not args.tsdf_pcd.exists():
        raise FileNotFoundError(f"TSDF point cloud not found: {args.tsdf_pcd}")
    if not args.openmvs_dense_cloud.exists():
        raise FileNotFoundError(f"OpenMVS dense cloud not found: {args.openmvs_dense_cloud}")

    thresholds = parse_thresholds(args.thresholds)
    output_dir = args.output_dir or default_output_dir(args.tsdf_pcd, workspace_dir)
    reset_output_dir(output_dir)

    print(f"Loading TSDF point cloud: {args.tsdf_pcd}")
    tsdf_raw = load_point_cloud(args.tsdf_pcd)
    print(f"Loading OpenMVS dense cloud: {args.openmvs_dense_cloud}")
    openmvs_raw = load_point_cloud(args.openmvs_dense_cloud)

    print("Exporting CloudCompare-friendly ASCII PLY files...")
    tsdf_ascii_ply = output_dir / "tsdf_input_standard_ascii.ply"
    openmvs_ascii_ply = output_dir / "openmvs_scene_dense_standard_ascii.ply"
    write_standard_ascii_ply(tsdf_raw, tsdf_ascii_ply)
    write_standard_ascii_ply(openmvs_raw, openmvs_ascii_ply)

    print("Preparing point clouds...")
    tsdf_prepared = prepare_cloud(
        tsdf_raw,
        voxel_size=args.voxel_size,
        normal_radius=args.normal_radius,
        max_points=args.max_points,
    )
    openmvs_prepared = prepare_cloud(
        openmvs_raw,
        voxel_size=args.voxel_size,
        normal_radius=args.normal_radius,
        max_points=args.max_points,
    )

    print("Computing no-GT consistency metrics...")
    metrics = compute_mutual_metrics(
        tsdf_prepared,
        openmvs_prepared,
        voxel_size=args.voxel_size,
        thresholds=thresholds,
    )
    workspace_info = load_workspace_info(workspace_dir)
    report = build_report(
        args,
        tsdf_raw,
        openmvs_raw,
        tsdf_prepared,
        openmvs_prepared,
        metrics,
        workspace_info,
    )
    openmvs_slam_metrics = compute_openmvs_slam_metrics(
        openmvs_prepared,
        workspace_info,
        thresholds=thresholds,
        frame_step=args.slam_eval_frame_step,
        max_frames=args.slam_eval_max_frames,
    )
    if openmvs_slam_metrics is not None:
        report["openmvs_slam_view_consistency"] = openmvs_slam_metrics

    report_json = output_dir / "comparison_report.json"
    report_txt = output_dir / "comparison_summary.txt"
    overview_png = output_dir / "overview_views.png"
    consistency_png = output_dir / "consistency_metrics.png"
    openmvs_slam_png = output_dir / "openmvs_slam_view_metrics.png"
    tsdf_eval_ply = output_dir / "tsdf_eval_downsampled_ascii.ply"
    openmvs_eval_ply = output_dir / "openmvs_eval_downsampled_ascii.ply"

    print("Saving figures...")
    plot_overview(get_points(tsdf_prepared), get_points(openmvs_prepared), overview_png)
    plot_consistency(metrics, thresholds, consistency_png)
    if openmvs_slam_metrics is not None:
        plot_openmvs_slam_metrics(openmvs_slam_metrics, openmvs_slam_png)

    print("Saving evaluation point clouds...")
    write_standard_ascii_ply(tsdf_prepared, tsdf_eval_ply)
    write_standard_ascii_ply(openmvs_prepared, openmvs_eval_ply)

    print("Saving reports...")
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_summary(report, report_txt)

    print("")
    print("Comparison finished.")
    print(f"Output dir: {output_dir}")
    print(f"Report JSON: {report_json}")
    print(f"Summary TXT: {report_txt}")
    print(f"Overview PNG: {overview_png}")
    print(f"Consistency PNG: {consistency_png}")
    if openmvs_slam_metrics is not None:
        print(f"OpenMVS SLAM PNG: {openmvs_slam_png}")
    print(f"TSDF ASCII PLY: {tsdf_ascii_ply}")
    print(f"OpenMVS ASCII PLY: {openmvs_ascii_ply}")
    print(f"TSDF eval PLY: {tsdf_eval_ply}")
    print(f"OpenMVS eval PLY: {openmvs_eval_ply}")
    print(
        "Key metrics: "
        f"Chamfer-L1={report['mutual_consistency']['symmetric_chamfer_l1_m']:.6f} m | "
        f"RMSE={report['mutual_consistency']['symmetric_rmse_m']:.6f} m | "
        f"Voxel IoU={report['mutual_consistency']['occupancy_iou']:.4f}"
    )
    if openmvs_slam_metrics is not None:
        print(
            "OpenMVS SLAM-view: "
            f"overlap={openmvs_slam_metrics['sensor_overlap_ratio']:.4f} | "
            f"MAE={0.0 if openmvs_slam_metrics['mae_m'] is None else openmvs_slam_metrics['mae_m']:.6f} m | "
            f"RMSE={0.0 if openmvs_slam_metrics['rmse_m'] is None else openmvs_slam_metrics['rmse_m']:.6f} m"
        )


if __name__ == "__main__":
    main()
