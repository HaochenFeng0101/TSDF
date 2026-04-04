#!/usr/bin/env python3
"""Visualize ModelNet40 samples from OFF or h5 files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import h5py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("h5py is required to read ModelNet40 h5 files.") from exc

try:
    import open3d as o3d
except Exception:
    o3d = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32, copy=False)
    centroid = points.mean(axis=0, keepdims=True)
    centered = points - centroid
    scale = np.linalg.norm(centered, axis=1).max()
    if scale > 0:
        centered = centered / scale
    return centered


def load_off_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
        if first_line == "OFF":
            counts_line = handle.readline().strip()
        elif first_line.startswith("OFF"):
            counts_line = first_line[3:].strip()
        else:
            raise ValueError(f"Invalid OFF file: {path}")

        while counts_line.startswith("#") or not counts_line:
            counts_line = handle.readline().strip()

        num_vertices, num_faces, _ = map(int, counts_line.split())

        vertices = []
        for _ in range(num_vertices):
            line = handle.readline().strip()
            while line.startswith("#") or not line:
                line = handle.readline().strip()
            vertices.append([float(v) for v in line.split()[:3]])

        faces = []
        for _ in range(num_faces):
            line = handle.readline().strip()
            while line.startswith("#") or not line:
                line = handle.readline().strip()
            parts = [int(v) for v in line.split()]
            corners = parts[1 : 1 + parts[0]]
            if len(corners) >= 3:
                for corner_idx in range(1, len(corners) - 1):
                    faces.append([corners[0], corners[corner_idx], corners[corner_idx + 1]])

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if len(vertices) == 0:
        raise ValueError(f"OFF file has no vertices: {path}")
    if len(faces) == 0:
        return vertices, faces

    finite_vertex_mask = np.isfinite(vertices).all(axis=1)
    if not np.any(finite_vertex_mask):
        raise ValueError(f"OFF file vertices are all invalid: {path}")

    if not np.all(finite_vertex_mask):
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[np.where(finite_vertex_mask)[0]] = np.arange(np.count_nonzero(finite_vertex_mask))
        vertices = vertices[finite_vertex_mask]
        valid_faces = np.all(finite_vertex_mask[faces], axis=1)
        faces = remap[faces[valid_faces]]

    valid_face_mask = ((faces >= 0).all(axis=1) & (faces < len(vertices)).all(axis=1))
    faces = faces[valid_face_mask]
    return vertices, faces


def sample_points_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
    sample_method: str,
) -> np.ndarray:
    safe_vertices = vertices[np.isfinite(vertices).all(axis=1)]
    if len(safe_vertices) == 0:
        raise ValueError("Mesh contains no finite vertices for sampling.")

    if sample_method == "vertex" or len(faces) == 0:
        replace = len(safe_vertices) < num_points
        indices = rng.choice(len(safe_vertices), num_points, replace=replace)
        return safe_vertices[indices].astype(np.float32)

    safe_vertices64 = safe_vertices.astype(np.float64, copy=False)
    scale = np.abs(safe_vertices64).max()
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    if scale > 1e6:
        safe_vertices64 = safe_vertices64 / scale

    triangles = safe_vertices64[faces]
    vec1 = triangles[:, 1] - triangles[:, 0]
    vec2 = triangles[:, 2] - triangles[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(vec1, vec2), axis=1)
    finite_area_mask = np.isfinite(areas) & (areas > 0)

    if not np.any(finite_area_mask):
        replace = len(safe_vertices) < num_points
        indices = rng.choice(len(safe_vertices), num_points, replace=replace)
        return safe_vertices[indices].astype(np.float32)

    triangles = triangles[finite_area_mask]
    areas = areas[finite_area_mask]
    probs = areas / areas.sum()
    face_indices = rng.choice(len(triangles), size=num_points, replace=True, p=probs)
    chosen = triangles[face_indices]

    r1 = np.sqrt(rng.random(num_points, dtype=np.float32))
    r2 = rng.random(num_points, dtype=np.float32)
    sampled = (
        (1.0 - r1)[:, None] * chosen[:, 0]
        + (r1 * (1.0 - r2))[:, None] * chosen[:, 1]
        + (r1 * r2)[:, None] * chosen[:, 2]
    )
    if scale > 1e6:
        sampled = sampled * scale
    return sampled.astype(np.float32)


def read_text_lines(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_label_names_near_h5(path: Path) -> list[str] | None:
    for candidate in (path.parent / "shape_names.txt", path.parent / "modelnet40_shape_names.txt", path.parent / "labels.txt"):
        if candidate.exists():
            labels = read_text_lines(candidate)
            if labels:
                return labels
    return None


def decode_string_values(values: np.ndarray) -> list[str]:
    decoded = []
    for value in values.reshape(-1):
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def normalize_path_string(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").lower()


def resolve_path_match(sample_paths: list[str], target_path: str) -> int:
    normalized_target = normalize_path_string(target_path)
    matched_indices = []
    for idx, sample_path in enumerate(sample_paths):
        normalized_sample = normalize_path_string(sample_path)
        if normalized_sample == normalized_target:
            matched_indices.append(idx)
            continue
        if normalized_sample.endswith(normalized_target) or normalized_target.endswith(normalized_sample):
            matched_indices.append(idx)

    if not matched_indices:
        raise ValueError(f"Could not find sample_path '{target_path}' in the h5 metadata.")
    return int(matched_indices[0])


def resolve_h5_sample(
    h5_path: Path,
    index: int,
    class_name: str | None,
    class_offset: int,
    sample_path: str | None,
    fallback_sample_path: str | None,
) -> dict[str, object]:
    with h5py.File(h5_path, "r") as handle:
        data = np.asarray(handle["data"], dtype=np.float32)[:, :, :3]
        labels = np.asarray(handle["label"]).reshape(-1).astype(np.int64)
        sample_paths = (
            decode_string_values(np.asarray(handle["sample_path"]))
            if "sample_path" in handle
            else None
        )
        stored_label_names = (
            decode_string_values(np.asarray(handle["label_name"]))
            if "label_name" in handle
            else None
        )

    label_lookup = load_label_names_near_h5(h5_path)
    label_names = []
    for item_idx, label_idx in enumerate(labels):
        if stored_label_names is not None and item_idx < len(stored_label_names):
            label_names.append(stored_label_names[item_idx])
        elif label_lookup is not None and 0 <= label_idx < len(label_lookup):
            label_names.append(label_lookup[label_idx])
        else:
            label_names.append(str(label_idx))

    if sample_path is not None:
        if sample_paths is None:
            raise ValueError(f"{h5_path} does not contain sample_path metadata.")
        resolved_index = resolve_path_match(sample_paths, sample_path)
    elif class_name is not None:
        normalized_class_name = class_name.strip().lower()
        matched = np.flatnonzero(
            np.asarray([name.lower() == normalized_class_name for name in label_names], dtype=bool)
        )
        if len(matched) == 0:
            raise ValueError(f"No samples with class '{class_name}' were found in {h5_path}.")
        if class_offset < 0 or class_offset >= len(matched):
            raise IndexError(
                f"class-offset {class_offset} is out of range for class '{class_name}' with {len(matched)} samples."
            )
        resolved_index = int(matched[class_offset])
    elif fallback_sample_path is not None and sample_paths is not None:
        resolved_index = resolve_path_match(sample_paths, fallback_sample_path)
    else:
        if index < 0 or index >= len(data):
            raise IndexError(f"Index {index} is out of range for {h5_path} with {len(data)} samples.")
        resolved_index = index

    return {
        "kind": "h5",
        "points": data[resolved_index],
        "label_idx": int(labels[resolved_index]),
        "label_name": label_names[resolved_index],
        "sample_path": sample_paths[resolved_index] if sample_paths is not None else None,
        "resolved_index": int(resolved_index),
        "num_samples": int(len(data)),
    }


def infer_off_label_name(path: Path) -> str:
    if path.parent.name in {"train", "test"}:
        return path.parent.parent.name
    return path.parent.name


def load_visual_source(
    path: Path,
    args: argparse.Namespace,
    fallback_sample_path: str | None = None,
) -> dict[str, object]:
    if path.suffix.lower() == ".off":
        rng = np.random.default_rng(args.seed)
        vertices, faces = load_off_mesh(path)
        points = sample_points_from_mesh(
            vertices=vertices,
            faces=faces,
            num_points=args.num_points,
            rng=rng,
            sample_method=args.sample_method,
        )
        return {
            "kind": "off",
            "points": points,
            "label_idx": None,
            "label_name": infer_off_label_name(path),
            "sample_path": str(path),
            "resolved_index": None,
            "num_samples": 1,
        }

    if path.suffix.lower() != ".h5":
        raise ValueError(f"Unsupported input type: {path}. Expected .off or .h5")

    return resolve_h5_sample(
        h5_path=path,
        index=args.index,
        class_name=args.class_name,
        class_offset=args.class_offset,
        sample_path=args.sample_path,
        fallback_sample_path=fallback_sample_path,
    )


def build_open3d_cloud(points: np.ndarray, color: tuple[float, float, float], offset_x: float):
    cloud = o3d.geometry.PointCloud()
    shifted = points[:, :3].astype(np.float64, copy=True)
    shifted[:, 0] += offset_x
    cloud.points = o3d.utility.Vector3dVector(shifted)
    cloud.paint_uniform_color(color)
    return cloud


def show_with_open3d(
    primary_points: np.ndarray,
    compare_points: np.ndarray | None,
    offset: float,
    point_size: float,
) -> None:
    geometries = [build_open3d_cloud(primary_points, color=(0.1, 0.55, 0.95), offset_x=0.0)]
    if compare_points is not None:
        geometries.append(build_open3d_cloud(compare_points, color=(0.95, 0.35, 0.15), offset_x=offset))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="ModelNet40 Viewer", width=1440, height=900)
    render_option = vis.get_render_option()
    render_option.point_size = float(point_size)
    render_option.background_color = np.array([0.03, 0.03, 0.03], dtype=np.float64)
    for geom in geometries:
        vis.add_geometry(geom)
    vis.run()
    vis.destroy_window()


def show_with_matplotlib(
    primary_points: np.ndarray,
    compare_points: np.ndarray | None,
    offset: float,
    title: str,
) -> None:
    if plt is None:
        raise RuntimeError("Neither open3d nor matplotlib is available for visualization.")

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="3d")
    p = primary_points[:, :3]
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=5, c="#2b8cfe", alpha=0.85, label="primary")

    if compare_points is not None:
        c = compare_points[:, :3].copy()
        c[:, 0] += offset
        ax.scatter(c[:, 0], c[:, 1], c[:, 2], s=5, c="#f05a28", alpha=0.8, label="compare")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show()


def maybe_normalize(points: np.ndarray, enabled: bool) -> np.ndarray:
    return normalize_points(points) if enabled else points.astype(np.float32, copy=False)


def describe_source(prefix: str, info: dict[str, object]) -> None:
    print(f"{prefix} kind   : {info['kind']}")
    print(f"{prefix} label  : {info['label_name']}")
    if info["label_idx"] is not None:
        print(f"{prefix} label_idx: {info['label_idx']}")
    if info["resolved_index"] is not None:
        print(f"{prefix} index  : {info['resolved_index']}")
    if info["sample_path"] is not None:
        print(f"{prefix} path   : {info['sample_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one or two ModelNet40 samples from OFF or h5 files.")
    parser.add_argument("--input", required=True, help="Primary .off or .h5 path.")
    parser.add_argument("--compare", default=None, help="Optional second .off or .h5 path.")
    parser.add_argument("--index", type=int, default=0, help="Sample index when reading h5.")
    parser.add_argument("--class-name", default=None, help="Optional class name when reading h5.")
    parser.add_argument(
        "--class-offset",
        type=int,
        default=0,
        help="When class-name is set, choose the N-th sample inside that class.",
    )
    parser.add_argument(
        "--sample-path",
        default=None,
        help="Resolve an h5 sample by stored sample_path metadata instead of index.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=2048,
        help="Point count used when sampling from OFF meshes.",
    )
    parser.add_argument(
        "--sample-method",
        default="surface",
        choices=["surface", "vertex"],
        help="How to sample points from OFF meshes.",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=1.6,
        help="X-axis offset used when compare is provided.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=4.0,
        help="Rendered point size for Open3D.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sampling seed used for OFF meshes.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable centering and scale normalization before display.",
    )
    args = parser.parse_args()

    primary_path = Path(args.input)
    primary_info = load_visual_source(primary_path, args)

    compare_info = None
    if args.compare:
        compare_path = Path(args.compare)
        fallback_sample_path = None
        if primary_info["kind"] == "off":
            fallback_sample_path = str(primary_path)
        elif primary_info["sample_path"] is not None:
            fallback_sample_path = str(primary_info["sample_path"])
        compare_info = load_visual_source(compare_path, args, fallback_sample_path=fallback_sample_path)

    display_normalize = not args.no_normalize
    primary_points = maybe_normalize(np.asarray(primary_info["points"], dtype=np.float32), display_normalize)
    compare_points = None
    if compare_info is not None:
        compare_points = maybe_normalize(np.asarray(compare_info["points"], dtype=np.float32), display_normalize)

    describe_source("Primary", primary_info)
    if compare_info is not None:
        describe_source("Compare", compare_info)
        print("Blue = primary, Red = compare (shifted along +X).")

    title = f"{primary_info['label_name']}"
    if primary_info["resolved_index"] is not None:
        title = f"Sample {primary_info['resolved_index']} | {title}"

    if o3d is not None:
        show_with_open3d(primary_points, compare_points, offset=args.offset, point_size=args.point_size)
    else:
        print("open3d is not installed, falling back to matplotlib.")
        show_with_matplotlib(primary_points, compare_points, offset=args.offset, title=title)


if __name__ == "__main__":
    main()
