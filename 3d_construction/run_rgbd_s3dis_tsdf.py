import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
TSDF_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_tum_rgbd_tsdf import (
    build_camera_model,
    default_output_path,
    load_config,
    load_tum_frames,
    resize_or_rectify_color,
    resize_or_rectify_depth,
)


def integrate_scene(args):
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
    return point_cloud


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Integrate a TUM RGB-D scene with ScalableTSDFVolume and save a single output point cloud."
    )
    parser.add_argument(
        "--config",
        default=str(TSDF_ROOT / "configs" / "rgbd" / "tum" / "fr3_office.yaml"),
        help="Path to a TUM RGB-D config file.",
    )
    parser.add_argument("--dataset", default=None, help="Override dataset path from the config file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output .pcd path. Defaults to 3d_construction/outputs/<config_stem>_s3dis_scene.pcd",
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


def main():
    parser = build_argparser()
    args = parser.parse_args()
    point_cloud = integrate_scene(args)

    output_path = Path(args.output) if args.output else default_output_path(
        args.config, "_s3dis_scene.pcd"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), point_cloud)
    print(f"Saved point cloud to {output_path}")


if __name__ == "__main__":
    main()
