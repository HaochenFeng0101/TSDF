import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
TSDF_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        cfg_special = yaml.full_load(handle)

    inherit_from = cfg_special.get("inherit_from")
    if inherit_from is not None:
        cfg = load_config(inherit_from)
    else:
        cfg = {}

    update_recursive(cfg, cfg_special)
    return cfg


def update_recursive(dict1, dict2):
    for key, value in dict2.items():
        if key not in dict1:
            dict1[key] = {}
        if isinstance(value, dict):
            update_recursive(dict1[key], value)
        else:
            dict1[key] = value


def parse_list(filepath, skiprows=0):
    return np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)


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


def load_tum_frames(dataset_path, frame_stride=1, max_frames=None, max_dt=0.08):
    if os.path.isfile(os.path.join(dataset_path, "groundtruth.txt")):
        pose_list = os.path.join(dataset_path, "groundtruth.txt")
    elif os.path.isfile(os.path.join(dataset_path, "pose.txt")):
        pose_list = os.path.join(dataset_path, "pose.txt")
    else:
        raise FileNotFoundError(
            f"Could not find groundtruth.txt or pose.txt in {dataset_path}"
        )

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

    associations = associate_frames(tstamp_image, tstamp_depth, tstamp_pose, max_dt)
    if frame_stride > 1:
        associations = associations[::frame_stride]
    if max_frames is not None:
        associations = associations[:max_frames]

    frames = []
    for image_idx, depth_idx, pose_idx in associations:
        quat = pose_vecs[pose_idx][4:]
        trans = pose_vecs[pose_idx][1:4]
        cam_to_world = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
        cam_to_world[:3, 3] = trans
        world_to_cam = np.linalg.inv(cam_to_world)

        frames.append(
            {
                "color_path": os.path.join(dataset_path, image_data[image_idx, 1]),
                "depth_path": os.path.join(dataset_path, depth_data[depth_idx, 1]),
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
    fx = calibration["fx"] * scale_x
    fy = calibration["fy"] * scale_y
    cx = calibration["cx"] * scale_x
    cy = calibration["cy"] * scale_y

    camera = {
        "raw_width": raw_width,
        "raw_height": raw_height,
        "width": target_width,
        "height": target_height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "depth_scale": calibration["depth_scale"],
        "distorted": calibration["distorted"],
        "dist_coeffs": np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ],
            dtype=np.float32,
        ),
    }
    camera["K_raw"] = np.array(
        [
            [calibration["fx"], 0.0, calibration["cx"]],
            [0.0, calibration["fy"], calibration["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    camera["K"] = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
    )

    if camera["distorted"]:
        map1x, map1y = cv2.initUndistortRectifyMap(
            camera["K_raw"],
            camera["dist_coeffs"],
            np.eye(3),
            camera["K"],
            (camera["width"], camera["height"]),
            cv2.CV_32FC1,
        )
        camera["map1x"] = map1x
        camera["map1y"] = map1y
    else:
        camera["map1x"] = None
        camera["map1y"] = None
    return camera


def resize_or_rectify_color(image, camera):
    if camera["distorted"]:
        return cv2.remap(image, camera["map1x"], camera["map1y"], cv2.INTER_LINEAR)
    if image.shape[1] == camera["width"] and image.shape[0] == camera["height"]:
        return image
    return cv2.resize(
        image,
        (camera["width"], camera["height"]),
        interpolation=cv2.INTER_LINEAR,
    )


def resize_or_rectify_depth(depth, camera):
    if camera["distorted"]:
        return cv2.remap(depth, camera["map1x"], camera["map1y"], cv2.INTER_NEAREST)
    if depth.shape[1] == camera["width"] and depth.shape[0] == camera["height"]:
        return depth
    return cv2.resize(
        depth,
        (camera["width"], camera["height"]),
        interpolation=cv2.INTER_NEAREST,
    )


def default_output_path(config_path, suffix):
    output_dir = TSDF_ROOT / "3d_construction" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{Path(config_path).stem}{suffix}"


def integrate_tsdf(args):
    config = load_config(args.config)
    dataset_path = args.dataset or config["Dataset"]["dataset_path"]
    frames = load_tum_frames(
        dataset_path,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        max_dt=args.max_dt,
    )
    if not frames:
        raise RuntimeError("No synchronized RGB-D frames were found for integration.")

    camera = build_camera_model(config, args.input_width, args.input_height)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        camera["width"],
        camera["height"],
        camera["fx"],
        camera["fy"],
        camera["cx"],
        camera["cy"],
    )
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        depth_sampling_stride=args.depth_sampling_stride,
    )

    for idx, frame in enumerate(frames, start=1):
        color = np.array(Image.open(frame["color_path"]))
        depth = np.array(Image.open(frame["depth_path"]))
        color = resize_or_rectify_color(color, camera)
        depth = resize_or_rectify_depth(depth, camera)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color.astype(np.uint8)),
            o3d.geometry.Image(depth.astype(np.uint16)),
            depth_scale=camera["depth_scale"],
            depth_trunc=args.depth_trunc,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, frame["extrinsic"])

        if idx % args.log_every == 0 or idx == len(frames):
            print(f"Integrated {idx}/{len(frames)} frames")

    point_cloud = volume.extract_point_cloud()
    if args.voxel_downsample > 0:
        point_cloud = point_cloud.voxel_down_sample(args.voxel_downsample)
    if args.remove_statistical_outlier:
        point_cloud, _ = point_cloud.remove_statistical_outlier(
            nb_neighbors=args.outlier_nb_neighbors,
            std_ratio=args.outlier_std_ratio,
        )

    output_path = Path(args.output) if args.output else default_output_path(
        args.config, ".pcd"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), point_cloud)
    print(f"Saved point cloud to {output_path}")

    if args.mesh_output:
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        mesh_output = Path(args.mesh_output)
        mesh_output.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(mesh_output), mesh)
        print(f"Saved mesh to {mesh_output}")


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Integrate a TUM RGB-D sequence with TSDF and save a point cloud."
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
        "--output",
        default=None,
        help="Output .pcd path. Defaults to TSDF/3d_construction/outputs/<config_stem>.pcd",
    )
    parser.add_argument(
        "--mesh-output",
        default=None,
        help="Optional output mesh path, e.g. TSDF/3d_construction/outputs/fr3_office_mesh.ply",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-dt", type=float, default=0.08)
    parser.add_argument("--input-width", type=int, default=None)
    parser.add_argument("--input-height", type=int, default=None)
    parser.add_argument("--voxel-length", type=float, default=0.01)
    parser.add_argument("--sdf-trunc", type=float, default=0.04)
    parser.add_argument("--depth-trunc", type=float, default=4.0)
    parser.add_argument("--depth-sampling-stride", type=int, default=4)
    parser.add_argument("--voxel-downsample", type=float, default=0.01)
    parser.add_argument("--remove-statistical-outlier", action="store_true")
    parser.add_argument("--outlier-nb-neighbors", type=int, default=20)
    parser.add_argument("--outlier-std-ratio", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=50)
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    integrate_tsdf(parser.parse_args())
