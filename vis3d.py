import argparse
import struct
from pathlib import Path

import numpy as np
import open3d as o3d
# python vis3d.py /home/haochen/code/TSDF/openmvs/workspaces/fr3_office_openmvs/scene_dense.ply \
#   --export /home/haochen/code/TSDF/openmvs/workspaces/fr3_office_openmvs/scene_dense_standard.ply


DEFAULT_PLY = Path("/home/haochen/code/TSDF/openmvs/workspaces/fr3_office_openmvs/scene_dense.ply")


def read_openmvs_ply(path: Path):
    with path.open("rb") as f:
        header = b""
        while b"end_header\n" not in header:
            chunk = f.read(1)
            if not chunk:
                raise ValueError(f"incomplete PLY header: {path}")
            header += chunk

        header_text = header.decode("ascii", errors="strict")
        if "format binary_little_endian 1.0" not in header_text:
            raise ValueError(f"unsupported PLY format in {path}")

        vertex_count = None
        for line in header_text.splitlines():
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
                break
        if vertex_count is None:
            raise ValueError(f"missing vertex count in {path}")

        fixed_fmt = struct.Struct("<fffBBBfff")
        points = np.empty((vertex_count, 3), dtype=np.float32)
        colors = np.empty((vertex_count, 3), dtype=np.float32)
        normals = np.empty((vertex_count, 3), dtype=np.float32)

        for i in range(vertex_count):
            raw = f.read(fixed_fmt.size)
            if len(raw) != fixed_fmt.size:
                raise ValueError(f"unexpected EOF while reading vertex {i} from {path}")
            x, y, z, r, g, b, nx, ny, nz = fixed_fmt.unpack(raw)
            points[i] = (x, y, z)
            colors[i] = (r / 255.0, g / 255.0, b / 255.0)
            normals[i] = (nx, ny, nz)

            num_view_indices_raw = f.read(1)
            if len(num_view_indices_raw) != 1:
                raise ValueError(f"unexpected EOF while reading view_indices count from {path}")
            num_view_indices = num_view_indices_raw[0]
            f.seek(num_view_indices * 4, 1)

            num_view_weights_raw = f.read(1)
            if len(num_view_weights_raw) != 1:
                raise ValueError(f"unexpected EOF while reading view_weights count from {path}")
            num_view_weights = num_view_weights_raw[0]
            f.seek(num_view_weights * 4, 1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return pcd


def load_point_cloud(path: Path):
    try:
        return read_openmvs_ply(path)
    except Exception:
        pcd = o3d.io.read_point_cloud(str(path))
        if len(pcd.points) == 0:
            raise
        return pcd


def main():
    parser = argparse.ArgumentParser(description="View OpenMVS point clouds in Open3D.")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_PLY), help="Path to a PLY point cloud.")
    parser.add_argument(
        "--export",
        help="Optional path to save a stripped standard PLY that tools like Open3D and CloudCompare can read.",
    )
    args = parser.parse_args()

    path = Path(args.path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"point cloud not found: {path}")

    pcd = load_point_cloud(path)
    print(f"Loaded {len(pcd.points)} points from {path}")

    if args.export:
        export_path = Path(args.export).expanduser().resolve()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_point_cloud(str(export_path), pcd):
            raise RuntimeError(f"failed to export point cloud to {export_path}")
        print(f"Exported standard PLY to {export_path}")

    o3d.visualization.draw_geometries([pcd], window_name="PLY Viewer", width=1280, height=800)


if __name__ == "__main__":
    main()
