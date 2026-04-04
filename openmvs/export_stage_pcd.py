#!/usr/bin/env python3
import argparse
from pathlib import Path

import open3d as o3d
'''
cd /home/haochen/code/TSDF
bash openmvs/run_openmvs_tum.sh \
  --config configs/rgbd/tum/fr3_office.yaml \
  --workspace-name fr3_office_openmvs_color \
  --reexport \
  --frame-stride 5 \
  --max-frames 2800 \
  --resolution-level 2 \
  --max-resolution 1600 \
  --min-resolution 640 \
  --number-views 4 \
  --number-views-fuse 2 \
  --iters 2 \
  --geometric-iters 1 \
  --free-space-support 0 \
  --mesh-close-holes 15 \
  --mesh-smooth 1 \
  --refine-scales 1 \
  --refine-max-views 4 \
  --stage-pcd-points 200000 \
  --max-threads 6



'''

def convert_ply_to_pcd(input_path: Path, output_path: Path, sample_points: int):
    cloud = o3d.io.read_point_cloud(str(input_path))
    if not cloud.is_empty():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(output_path), cloud)
        return "point_cloud"

    mesh = o3d.io.read_triangle_mesh(str(input_path))
    if mesh.is_empty():
        raise RuntimeError(f"Could not load a point cloud or mesh from: {input_path}")

    if sample_points > 0:
        cloud = mesh.sample_points_uniformly(number_of_points=sample_points)
    else:
        cloud = o3d.geometry.PointCloud()
        cloud.points = mesh.vertices
        if mesh.has_vertex_colors():
            cloud.colors = mesh.vertex_colors
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), cloud)
    return "mesh_sampled"


def main():
    parser = argparse.ArgumentParser(
        description="Convert an OpenMVS stage .ply file to .pcd. Meshes are uniformly sampled."
    )
    parser.add_argument("input", help="Input .ply path.")
    parser.add_argument("output", help="Output .pcd path.")
    parser.add_argument(
        "--sample-points",
        type=int,
        default=500000,
        help="Number of points to sample when the input is a mesh. Use 0 to export vertices only.",
    )
    args = parser.parse_args()

    mode = convert_ply_to_pcd(Path(args.input), Path(args.output), args.sample_points)
    print(f"Saved {mode} PCD to {args.output}")


if __name__ == "__main__":
    main()
