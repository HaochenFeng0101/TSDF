import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
TSDF_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def update_recursive(dict1, dict2):
    for key, value in dict2.items():
        if key not in dict1:
            dict1[key] = {}
        if isinstance(value, dict):
            update_recursive(dict1[key], value)
        else:
            dict1[key] = value


def load_intrinsic_matrix(path):
    matrix = np.loadtxt(path, dtype=np.float32)
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected 3x3 or 4x4 intrinsic matrix in {path}, got {matrix.shape}")
    return matrix


def sorted_frame_map(folder, suffixes):
    mapping = {}
    for suffix in suffixes:
        for path in folder.glob(f"*{suffix}"):
            mapping[path.stem] = path
    return dict(sorted(mapping.items(), key=lambda item: int(item[0])))


def load_scannet_frames(scene_path, frame_stride=1, max_frames=None):
    scene_path = Path(scene_path)
    color_dir = scene_path / "color"
    depth_dir = scene_path / "depth"
    pose_dir = scene_path / "pose"
    intrinsic_dir = scene_path / "intrinsic"

    for folder in [color_dir, depth_dir, pose_dir, intrinsic_dir]:
        if not folder.exists():
            raise FileNotFoundError(
                f"Expected ScanNet exported folder {folder}. "
                "Use dataset/download_scannet_scene.py with --reader-script first."
            )

    color_map = sorted_frame_map(color_dir, [".jpg", ".png"])
    depth_map = sorted_frame_map(depth_dir, [".png"])
    pose_map = sorted_frame_map(pose_dir, [".txt"])

    common_ids = [frame_id for frame_id in color_map if frame_id in depth_map and frame_id in pose_map]
    if frame_stride > 1:
        common_ids = common_ids[::frame_stride]
    if max_frames is not None:
        common_ids = common_ids[:max_frames]

    frames = []
    for frame_id in common_ids:
        pose = np.loadtxt(pose_map[frame_id], dtype=np.float32)
        if pose.shape != (4, 4):
            continue
        if not np.isfinite(pose).all():
            continue
        if abs(np.linalg.det(pose[:3, :3])) < 1e-8:
            continue

        world_to_cam = np.linalg.inv(pose)
        frames.append(
            {
                "frame_id": frame_id,
                "color_path": color_map[frame_id],
                "depth_path": depth_map[frame_id],
                "extrinsic": world_to_cam,
            }
        )
    return frames


def build_camera_model(scene_path, config, input_width=None, input_height=None):
    scene_path = Path(scene_path)
    intrinsic_dir = scene_path / "intrinsic"
    intrinsic_path = intrinsic_dir / "intrinsic_depth.txt"
    if not intrinsic_path.exists():
        intrinsic_path = intrinsic_dir / "intrinsic_color.txt"
    if not intrinsic_path.exists():
        raise FileNotFoundError(
            f"Could not find intrinsic_depth.txt or intrinsic_color.txt in {intrinsic_dir}"
        )

    depth_dir = scene_path / "depth"
    first_depth = sorted(depth_dir.glob("*.png"))
    if not first_depth:
        raise FileNotFoundError(f"No depth png files found in {depth_dir}")
    raw_depth = np.array(Image.open(first_depth[0]))
    raw_height, raw_width = raw_depth.shape[:2]

    intrinsic = load_intrinsic_matrix(intrinsic_path)
    target_width = input_width or config["Dataset"].get("input_width", raw_width)
    target_height = input_height or config["Dataset"].get("input_height", raw_height)
    scale_x = target_width / raw_width
    scale_y = target_height / raw_height

    fx = float(intrinsic[0, 0] * scale_x)
    fy = float(intrinsic[1, 1] * scale_y)
    cx = float(intrinsic[0, 2] * scale_x)
    cy = float(intrinsic[1, 2] * scale_y)

    return {
        "raw_width": raw_width,
        "raw_height": raw_height,
        "width": target_width,
        "height": target_height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "depth_scale": float(config["Dataset"]["Calibration"].get("depth_scale", 1000.0)),
    }


def resize_color(image, camera):
    if image.shape[1] == camera["width"] and image.shape[0] == camera["height"]:
        return image
    return cv2.resize(
        image,
        (camera["width"], camera["height"]),
        interpolation=cv2.INTER_LINEAR,
    )


def resize_depth(depth, camera):
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
    scene_path = args.dataset or config["Dataset"]["dataset_path"]
    frames = load_scannet_frames(
        scene_path,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )
    if not frames:
        raise RuntimeError("No valid ScanNet RGB-D frames were found for integration.")

    camera = build_camera_model(scene_path, config, args.input_width, args.input_height)
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
        color = resize_color(color, camera)
        depth = resize_depth(depth, camera)

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
        description="Integrate a ScanNet exported RGB-D scene with TSDF and save a point cloud."
    )
    parser.add_argument(
        "--config",
        default=str(TSDF_ROOT / "configs" / "rgbd" / "scannet" / "scene0000_00.yaml"),
        help="Path to a ScanNet config file.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override exported ScanNet scene path from the config file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .pcd path. Defaults to TSDF/3d_construction/outputs/<config_stem>.pcd",
    )
    parser.add_argument(
        "--mesh-output",
        default=None,
        help="Optional output mesh path.",
    )
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--input-width", type=int, default=None)
    parser.add_argument("--input-height", type=int, default=None)
    parser.add_argument("--voxel-length", type=float, default=0.02)
    parser.add_argument("--sdf-trunc", type=float, default=0.08)
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
