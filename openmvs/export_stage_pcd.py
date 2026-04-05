#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

def sample_points(points: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) >= num_points:
        indices = rng.choice(len(points), num_points, replace=False)
    else:
        indices = rng.choice(len(points), num_points, replace=True)
    return points[indices]


def sample_textured_mesh(mesh: o3d.geometry.TriangleMesh, sample_points_count: int):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if len(vertices) == 0 or len(triangles) == 0:
        raise RuntimeError("Mesh is empty.")

    triangle_vertices = vertices[triangles]
    areas = 0.5 * np.linalg.norm(
        np.cross(
            triangle_vertices[:, 1] - triangle_vertices[:, 0],
            triangle_vertices[:, 2] - triangle_vertices[:, 0],
        ),
        axis=1,
    )
    valid = np.isfinite(areas) & (areas > 0)
    if not np.any(valid):
        raise RuntimeError("Mesh has no valid triangles for sampling.")

    triangles = triangles[valid]
    triangle_vertices = triangle_vertices[valid]
    areas = areas[valid]
    probs = areas / areas.sum()

    rng = np.random.default_rng(0)
    face_indices = rng.choice(len(triangle_vertices), size=sample_points_count, replace=True, p=probs)
    chosen_triangles = triangle_vertices[face_indices]

    r1 = np.sqrt(rng.random(sample_points_count))
    r2 = rng.random(sample_points_count)
    w0 = 1.0 - r1
    w1 = r1 * (1.0 - r2)
    w2 = r1 * r2
    sampled_points = (
        w0[:, None] * chosen_triangles[:, 0]
        + w1[:, None] * chosen_triangles[:, 1]
        + w2[:, None] * chosen_triangles[:, 2]
    ).astype(np.float64)

    sampled_colors = None

    if mesh.has_triangle_uvs() and len(mesh.textures) > 0:
        triangle_uvs = np.asarray(mesh.triangle_uvs, dtype=np.float64).reshape(-1, 3, 2)
        triangle_uvs = triangle_uvs[valid]
        chosen_uvs = triangle_uvs[face_indices]
        sampled_uvs = (
            w0[:, None] * chosen_uvs[:, 0]
            + w1[:, None] * chosen_uvs[:, 1]
            + w2[:, None] * chosen_uvs[:, 2]
        )

        texture_ids = None
        if hasattr(mesh, "triangle_material_ids"):
            triangle_material_ids = np.asarray(mesh.triangle_material_ids, dtype=np.int64)
            if len(triangle_material_ids) == len(valid):
                triangle_material_ids = triangle_material_ids[valid]
                texture_ids = triangle_material_ids[face_indices]
        if texture_ids is None or len(texture_ids) != sample_points_count:
            texture_ids = np.zeros(sample_points_count, dtype=np.int64)

        sampled_colors = np.zeros((sample_points_count, 3), dtype=np.float64)
        texture_arrays = [np.asarray(texture) for texture in mesh.textures]
        for texture_idx, texture_array in enumerate(texture_arrays):
            selection = texture_ids == texture_idx
            if not np.any(selection):
                continue
            image = texture_array
            if image.ndim == 2:
                image = np.repeat(image[:, :, None], 3, axis=2)
            image = image[:, :, :3].astype(np.float64)
            height, width = image.shape[:2]
            uvs = sampled_uvs[selection]
            u = np.clip(uvs[:, 0], 0.0, 1.0)
            v = np.clip(uvs[:, 1], 0.0, 1.0)
            x = np.clip(np.round(u * (width - 1)).astype(np.int64), 0, width - 1)
            y = np.clip(np.round((1.0 - v) * (height - 1)).astype(np.int64), 0, height - 1)
            sampled_colors[selection] = image[y, x] / 255.0

    elif mesh.has_vertex_colors():
        vertex_colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
        chosen_vertex_indices = triangles[face_indices]
        chosen_vertex_colors = vertex_colors[chosen_vertex_indices]
        sampled_colors = (
            w0[:, None] * chosen_vertex_colors[:, 0]
            + w1[:, None] * chosen_vertex_colors[:, 1]
            + w2[:, None] * chosen_vertex_colors[:, 2]
        )

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(sampled_points)
    if sampled_colors is not None and len(sampled_colors) == len(sampled_points):
        cloud.colors = o3d.utility.Vector3dVector(np.clip(sampled_colors, 0.0, 1.0))
    return cloud


def convert_ply_to_pcd(input_path: Path, output_path: Path, sample_points_count: int):
    cloud = o3d.io.read_point_cloud(str(input_path))
    if not cloud.is_empty():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(output_path), cloud)
        return "point_cloud", cloud.has_colors()

    mesh = o3d.io.read_triangle_mesh(str(input_path), enable_post_processing=True)
    if mesh.is_empty():
        raise RuntimeError(f"Could not load a point cloud or mesh from: {input_path}")

    if sample_points_count > 0:
        if mesh.has_triangle_uvs() and len(mesh.textures) > 0:
            cloud = sample_textured_mesh(mesh, sample_points_count)
        else:
            cloud = mesh.sample_points_uniformly(number_of_points=sample_points_count)
    else:
        cloud = o3d.geometry.PointCloud()
        cloud.points = mesh.vertices
        if mesh.has_vertex_colors():
            cloud.colors = mesh.vertex_colors
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), cloud)
    return "mesh_sampled", cloud.has_colors()


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

    mode, has_color = convert_ply_to_pcd(Path(args.input), Path(args.output), args.sample_points)
    color_note = "with color" if has_color else "without color"
    print(f"Saved {mode} PCD {color_note} to {args.output}")


if __name__ == "__main__":
    main()
