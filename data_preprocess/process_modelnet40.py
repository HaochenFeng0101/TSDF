#!/usr/bin/env python3
"""Apply mild point corruption to ModelNet40 and export processed splits.

The script accepts either:
- the original OFF-mesh directory layout
- an existing ModelNet40 h5 directory such as modelnet40_ply_hdf5_2048

By default, output is written in a ModelNet40-style class/split directory layout:
- <output>/<class>/train/*.npy
- <output>/<class>/test/*.npy

Metadata files are also written:
- shape_names.txt
- modelnet40_shape_names.txt
- labels.txt
- modelnet40_train.txt / modelnet40_test.txt

Optional legacy h5 export is still supported for backward compatibility.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

try:
    import h5py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("h5py is required to process ModelNet40 h5 files.") from exc


POINT_EXTENSIONS = (".off", ".npy", ".npz", ".txt", ".pts", ".xyz")


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32, copy=False)
    centroid = points.mean(axis=0, keepdims=True)
    centered = points - centroid
    scale = np.linalg.norm(centered, axis=1).max()
    if scale > 0:
        centered = centered / scale
    return centered


def apply_single_corruption(
    points: np.ndarray,
    reference_points: np.ndarray,
    rng: np.random.Generator,
    random_drop_min: float,
    random_drop_max: float,
    local_drop_min: float,
    local_drop_max: float,
    min_keep_ratio: float,
) -> tuple[np.ndarray, dict[str, float]]:
    num_points = len(points)
    work_points = points.astype(np.float32, copy=True)
    distance_points = reference_points.astype(np.float32, copy=False)

    random_drop_ratio = float(rng.uniform(random_drop_min, random_drop_max))
    local_drop_ratio = float(rng.uniform(local_drop_min, local_drop_max))
    corruption_type = "random_dropout" if rng.random() < 0.5 else "local_dropout"

    keep_mask = np.ones(num_points, dtype=bool)

    if corruption_type == "random_dropout":
        if random_drop_ratio > 0:
            random_drop = rng.random(num_points) < random_drop_ratio
            keep_mask[random_drop] = False
    else:
        current_keep = np.flatnonzero(keep_mask)
        if local_drop_ratio > 0 and len(current_keep) > 8:
            anchor = int(rng.choice(current_keep))
            target_local_drop = max(1, int(round(num_points * local_drop_ratio)))
            local_drop = min(target_local_drop, max(1, len(current_keep) - 8))
            if local_drop > 0:
                distances = np.linalg.norm(distance_points - distance_points[anchor], axis=1)
                nearest = np.argsort(distances)
                local_candidates = nearest[keep_mask[nearest]]
                keep_mask[local_candidates[:local_drop]] = False

    keep_indices = np.flatnonzero(keep_mask)
    min_keep = max(8, int(round(num_points * min_keep_ratio)))
    if len(keep_indices) < min_keep:
        restore_count = min_keep - len(keep_indices)
        restore_candidates = np.flatnonzero(~keep_mask)
        if restore_count > 0 and len(restore_candidates) > 0:
            restore = rng.choice(
                restore_candidates,
                size=min(restore_count, len(restore_candidates)),
                replace=False,
            )
            keep_mask[restore] = True
            keep_indices = np.flatnonzero(keep_mask)

    kept = work_points[keep_indices]
    if len(kept) == 0:
        kept = work_points[:1]

    if len(kept) < num_points:
        duplicate_idx = rng.choice(len(kept), size=num_points - len(kept), replace=True)
        duplicated = kept[duplicate_idx]
        corrupted = np.concatenate([kept, duplicated], axis=0)
    elif len(kept) > num_points:
        sample_idx = rng.choice(len(kept), size=num_points, replace=False)
        corrupted = kept[sample_idx]
    else:
        corrupted = kept

    corrupted = corrupted[rng.permutation(num_points)].astype(np.float32, copy=False)
    stats = {
        "corruption_type": corruption_type,
        "random_drop_ratio": random_drop_ratio if corruption_type == "random_dropout" else 0.0,
        "local_drop_ratio": local_drop_ratio if corruption_type == "local_dropout" else 0.0,
        "kept_ratio_before_refill": float(len(keep_indices) / num_points),
        "final_unique_ratio": float(len(np.unique(corrupted, axis=0)) / num_points),
    }
    return corrupted, stats


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

    finite_vertex_mask = np.isfinite(vertices).all(axis=1)
    if not np.any(finite_vertex_mask):
        raise ValueError(f"OFF file vertices are all invalid: {path}")

    if not np.all(finite_vertex_mask):
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[np.where(finite_vertex_mask)[0]] = np.arange(np.count_nonzero(finite_vertex_mask))
        vertices = vertices[finite_vertex_mask]
        if len(faces) > 0:
            valid_faces = np.all(finite_vertex_mask[faces], axis=1)
            faces = remap[faces[valid_faces]]

    if len(faces) > 0:
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


def load_point_cloud_file(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        key = "points" if "points" in data else list(data.keys())[0]
        points = data[key]
    elif suffix in {".txt", ".pts", ".xyz"}:
        points = np.loadtxt(path)
    else:
        raise ValueError(f"Unsupported point file format: {path}")

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected Nx3+ points in {path}, got shape {points.shape}")
    return points[:, :3]


def _looks_like_modelnet_off_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False

    split_files = [path / "modelnet40_train.txt", path / "modelnet40_test.txt"]
    if any(candidate.exists() for candidate in split_files):
        return True

    shape_name_files = [
        path / "modelnet40_shape_names.txt",
        path / "shape_names.txt",
        path / "labels.txt",
    ]
    if any(candidate.exists() for candidate in shape_name_files):
        return True

    for class_dir in path.iterdir():
        if not class_dir.is_dir():
            continue
        if (class_dir / "train").exists() or (class_dir / "test").exists():
            return True
        if any(class_dir.glob("*.off")):
            return True
    return False


def resolve_modelnet_off_root(root: Path) -> Path:
    root = Path(root)
    candidates = [root, root / "ModelNet40", root / "modelnet40"]
    for candidate in candidates:
        if _looks_like_modelnet_off_root(candidate):
            return candidate

    for candidate in candidates:
        if not candidate.exists():
            continue
        for subdir in sorted(path for path in candidate.rglob("*") if path.is_dir()):
            if _looks_like_modelnet_off_root(subdir):
                return subdir
    raise FileNotFoundError(f"Could not find ModelNet40 OFF root under {root}")


def resolve_modelnet_h5_root(root: Path) -> Path:
    root = Path(root)
    candidates = [root, root / "modelnet40_ply_hdf5_2048"]
    for candidate in candidates:
        if (candidate / "train_files.txt").exists() and (candidate / "shape_names.txt").exists():
            return candidate

    for subdir in sorted(path for path in root.rglob("*") if path.is_dir()):
        if (subdir / "train_files.txt").exists() and (subdir / "shape_names.txt").exists():
            return subdir
    raise FileNotFoundError(f"Could not find ModelNet40 h5 root under {root}")


def read_text_lines(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def read_shape_names(root: Path) -> list[str]:
    for candidate in (root / "shape_names.txt", root / "modelnet40_shape_names.txt", root / "labels.txt"):
        if candidate.exists():
            labels = read_text_lines(candidate)
            if labels:
                return labels

    labels = sorted(path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
    if not labels:
        raise ValueError(f"No class labels found under {root}")
    return labels


def scan_split_dirs(root: Path, split: str, labels: list[str]) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for label in labels:
        split_dir = root / label / split
        if split_dir.exists():
            for path in sorted(split_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in POINT_EXTENSIONS:
                    samples.append({"path": str(path), "label": label})
            continue

        label_dir = root / label
        if not label_dir.exists():
            continue
        for path in sorted(label_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in POINT_EXTENSIONS and split in path.stem:
                samples.append({"path": str(path), "label": label})
    return samples


def find_sample_path(root: Path, label: str, stem: str) -> Path:
    candidates = []
    for suffix in POINT_EXTENSIONS:
        candidates.extend(
            [
                root / label / f"{stem}{suffix}",
                root / label / "train" / f"{stem}{suffix}",
                root / label / "test" / f"{stem}{suffix}",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate ModelNet40 sample for {label}/{stem}")


def read_split_list(root: Path, split: str, labels: list[str]) -> list[dict[str, str]] | None:
    split_candidates = [root / f"modelnet40_{split}.txt", root / f"{split}_files.txt"]
    split_file = next((candidate for candidate in split_candidates if candidate.exists()), None)
    if split_file is None:
        return None

    label_set = set(labels)
    samples: list[dict[str, str]] = []
    for item in read_text_lines(split_file):
        stem = Path(item.replace("\\", "/")).stem
        matched_label = None
        for label in label_set:
            if stem.startswith(f"{label}_"):
                matched_label = label
                break
        if matched_label is None:
            continue
        actual_stem = stem[len(matched_label) + 1 :]
        samples.append({"path": str(find_sample_path(root, matched_label, actual_stem)), "label": matched_label})
    return samples


def build_modelnet40_splits(root: Path) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    root = resolve_modelnet_off_root(root)
    labels = read_shape_names(root)
    train_samples = read_split_list(root, "train", labels)
    test_samples = read_split_list(root, "test", labels)

    if train_samples is None or test_samples is None:
        train_samples = scan_split_dirs(root, "train", labels)
        test_samples = scan_split_dirs(root, "test", labels)

    if not train_samples or not test_samples:
        raise ValueError(
            f"Could not build ModelNet40 train/test splits under {root}. "
            "Expected split files or class/train|test folders."
        )
    return labels, train_samples, test_samples


def decode_string_values(values: np.ndarray) -> list[str]:
    decoded = []
    for value in values.reshape(-1):
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def resolve_h5_file(root: Path, item: str) -> Path:
    relpath = Path(item.replace("\\", "/"))
    candidates = [root / relpath.name, root / relpath]
    if relpath.is_absolute():
        candidates.insert(0, relpath)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find H5 file listed in split file: {item}")


def load_h5_split_samples(root: Path, split: str) -> tuple[list[str], list[dict[str, object]]]:
    root = resolve_modelnet_h5_root(root)
    labels = read_shape_names(root)
    filelist_name = "train_files.txt" if split == "train" else "test_files.txt"
    filelist_path = root / filelist_name
    if not filelist_path.exists():
        raise FileNotFoundError(f"Could not find {filelist_path}")

    samples: list[dict[str, object]] = []
    for item in read_text_lines(filelist_path):
        h5_path = resolve_h5_file(root, item)
        with h5py.File(h5_path, "r") as handle:
            data = np.asarray(handle["data"], dtype=np.float32)[:, :, :3]
            label_values = np.asarray(handle["label"]).reshape(-1).astype(np.int64)
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

        for sample_idx, label_idx in enumerate(label_values):
            if stored_label_names is not None and sample_idx < len(stored_label_names):
                label_name = stored_label_names[sample_idx]
            elif 0 <= label_idx < len(labels):
                label_name = labels[label_idx]
            else:
                label_name = str(label_idx)

            sample_path = (
                sample_paths[sample_idx]
                if sample_paths is not None and sample_idx < len(sample_paths)
                else f"{h5_path.name}#{sample_idx}"
            )

            samples.append(
                {
                    "points": data[sample_idx],
                    "label_idx": int(label_idx),
                    "label_name": label_name,
                    "sample_path": sample_path,
                }
            )
    return labels, samples


def write_output_metadata(output_dir: Path, labels: list[str]) -> None:
    shape_names_path = output_dir / "shape_names.txt"
    modelnet_shape_names_path = output_dir / "modelnet40_shape_names.txt"
    labels_path = output_dir / "labels.txt"

    for target in (shape_names_path, modelnet_shape_names_path, labels_path):
        with open(target, "w", encoding="utf-8") as handle:
            for label in labels:
                handle.write(f"{label}\n")


def write_split_file(output_dir: Path, split: str, h5_filename: str) -> None:
    target = output_dir / ("train_files.txt" if split == "train" else "test_files.txt")
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(f"{h5_filename}\n")


def slugify_name(value: str) -> str:
    text = value.replace("\\", "/").strip()
    if not text:
        return "sample"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "sample"


def infer_sample_stem(sample: dict[str, object], fallback_index: int) -> str:
    sample_path = str(sample.get("sample_path", "")).replace("\\", "/")
    if sample_path:
        if "#" in sample_path:
            base, suffix = sample_path.split("#", 1)
            stem = f"{Path(base).stem}_{slugify_name(suffix)}"
        else:
            stem = Path(sample_path).stem
        stem = slugify_name(stem)
        if stem:
            return stem
    return f"sample_{fallback_index:05d}"


def process_single_sample(
    sample: dict[str, object],
    rng: np.random.Generator,
    random_drop_min: float,
    random_drop_max: float,
    local_drop_min: float,
    local_drop_max: float,
    min_keep_ratio: float,
    normalize_before_chunk: bool,
    off_num_points: int,
    sample_method: str,
) -> tuple[np.ndarray, int, str, str, dict[str, float]]:
    if "points" in sample:
        points = np.asarray(sample["points"], dtype=np.float32)
    else:
        sample_path = Path(str(sample["path"]))
        if sample_path.suffix.lower() == ".off":
            vertices, faces = load_off_mesh(sample_path)
            points = sample_points_from_mesh(
                vertices=vertices,
                faces=faces,
                num_points=off_num_points,
                rng=rng,
                sample_method=sample_method,
            )
        else:
            points = load_point_cloud_file(sample_path)

    reference_points = normalize_points(points) if normalize_before_chunk else points
    corrupted, stats = apply_single_corruption(
        points=points,
        reference_points=reference_points,
        rng=rng,
        random_drop_min=random_drop_min,
        random_drop_max=random_drop_max,
        local_drop_min=local_drop_min,
        local_drop_max=local_drop_max,
        min_keep_ratio=min_keep_ratio,
    )
    return (
        corrupted,
        int(sample["label_idx"]),
        str(sample["sample_path"]),
        str(sample["label_name"]),
        stats,
    )


def summarize_stats(stats_list: list[dict[str, float]]) -> dict[str, float]:
    random_drop_values = [stats["random_drop_ratio"] for stats in stats_list]
    local_drop_values = [stats["local_drop_ratio"] for stats in stats_list]
    keep_values = [stats["kept_ratio_before_refill"] for stats in stats_list]
    unique_values = [stats["final_unique_ratio"] for stats in stats_list]
    random_mode_count = sum(1 for stats in stats_list if stats["corruption_type"] == "random_dropout")
    local_mode_count = len(stats_list) - random_mode_count
    return {
        "mean_random_drop_ratio": float(np.mean(random_drop_values)) if random_drop_values else 0.0,
        "mean_local_drop_ratio": float(np.mean(local_drop_values)) if local_drop_values else 0.0,
        "mean_kept_ratio_before_refill": float(np.mean(keep_values)) if keep_values else 0.0,
        "mean_final_unique_ratio": float(np.mean(unique_values)) if unique_values else 0.0,
        "random_mode_count": float(random_mode_count),
        "local_mode_count": float(local_mode_count),
    }


def process_samples(
    samples: list[dict[str, object]],
    seed: int,
    random_drop_min: float,
    random_drop_max: float,
    local_drop_min: float,
    local_drop_max: float,
    min_keep_ratio: float,
    normalize_before_chunk: bool,
    off_num_points: int,
    sample_method: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, float]]:
    rng = np.random.default_rng(seed)
    processed_data: list[np.ndarray] = []
    label_values: list[int] = []
    sample_paths: list[str] = []
    label_names: list[str] = []
    stats_list: list[dict[str, float]] = []

    for sample in samples:
        corrupted, label_idx, sample_path, label_name, stats = process_single_sample(
            sample=sample,
            rng=rng,
            random_drop_min=random_drop_min,
            random_drop_max=random_drop_max,
            local_drop_min=local_drop_min,
            local_drop_max=local_drop_max,
            min_keep_ratio=min_keep_ratio,
            normalize_before_chunk=normalize_before_chunk,
            off_num_points=off_num_points,
            sample_method=sample_method,
        )
        processed_data.append(corrupted)
        label_values.append(label_idx)
        sample_paths.append(sample_path)
        label_names.append(label_name)
        stats_list.append(stats)

    summary = summarize_stats(stats_list)
    return (
        np.stack(processed_data, axis=0).astype(np.float32),
        np.asarray(label_values, dtype=np.int64).reshape(-1, 1),
        sample_paths,
        label_names,
        summary,
    )

def infer_input_format(root: Path) -> str:
    try:
        resolve_modelnet_h5_root(root)
        return "h5"
    except Exception:
        pass

    try:
        resolve_modelnet_off_root(root)
        return "off"
    except Exception:
        pass

    raise FileNotFoundError(f"Could not detect a usable ModelNet40 dataset under {root}")


def build_directory_records(
    split_samples: list[dict[str, object]],
    processed_data: np.ndarray,
) -> list[dict[str, object]]:
    used_names: set[str] = set()
    records: list[dict[str, object]] = []

    for index, (sample, points) in enumerate(zip(split_samples, processed_data)):
        label_name = str(sample["label_name"])
        stem_base = infer_sample_stem(sample, index)
        stem = stem_base
        duplicate_index = 1
        key = f"{label_name}/{stem}"
        while key in used_names:
            stem = f"{stem_base}_{duplicate_index:02d}"
            duplicate_index += 1
            key = f"{label_name}/{stem}"
        used_names.add(key)
        records.append({"label_name": label_name, "stem": stem, "points": points})
    return records


def write_modelnet_directory_split(
    output_dir: Path,
    split: str,
    labels: list[str],
    records: list[dict[str, object]],
    overwrite: bool,
) -> list[str]:
    manifest_entries: list[str] = []
    for label in labels:
        (output_dir / label / split).mkdir(parents=True, exist_ok=True)

    for record in records:
        label_name = str(record["label_name"])
        stem = str(record["stem"])
        target = output_dir / label_name / split / f"{stem}.npy"
        if target.exists() and not overwrite:
            raise FileExistsError(f"Output file already exists: {target}. Use --overwrite to replace it.")
        np.save(target, np.asarray(record["points"], dtype=np.float32))
        manifest_entries.append(f"{label_name}_{stem}")

    manifest_path = output_dir / f"modelnet40_{split}.txt"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for entry in manifest_entries:
            handle.write(f"{entry}\n")
    return manifest_entries



def process_modelnet_directory_split(
    output_dir: Path,
    split: str,
    labels: list[str],
    samples: list[dict[str, object]],
    seed: int,
    random_drop_min: float,
    random_drop_max: float,
    local_drop_min: float,
    local_drop_max: float,
    min_keep_ratio: float,
    normalize_before_chunk: bool,
    off_num_points: int,
    sample_method: str,
    overwrite: bool,
    progress_every: int = 200,
) -> tuple[int, dict[str, float]]:
    rng = np.random.default_rng(seed)
    for label in labels:
        (output_dir / label / split).mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    manifest_entries: list[str] = []
    stats_list: list[dict[str, float]] = []

    total = len(samples)
    for index, sample in enumerate(samples, start=1):
        corrupted, _label_idx, _sample_path, label_name, stats = process_single_sample(
            sample=sample,
            rng=rng,
            random_drop_min=random_drop_min,
            random_drop_max=random_drop_max,
            local_drop_min=local_drop_min,
            local_drop_max=local_drop_max,
            min_keep_ratio=min_keep_ratio,
            normalize_before_chunk=normalize_before_chunk,
            off_num_points=off_num_points,
            sample_method=sample_method,
        )
        stem_base = infer_sample_stem(sample, index - 1)
        stem = stem_base
        duplicate_index = 1
        key = f"{label_name}/{stem}"
        while key in used_names:
            stem = f"{stem_base}_{duplicate_index:02d}"
            duplicate_index += 1
            key = f"{label_name}/{stem}"
        used_names.add(key)

        target = output_dir / label_name / split / f"{stem}.npy"
        if target.exists() and not overwrite:
            raise FileExistsError(f"Output file already exists: {target}. Use --overwrite to replace it.")
        np.save(target, corrupted.astype(np.float32))
        manifest_entries.append(f"{label_name}_{stem}")
        stats_list.append(stats)

        if index == 1 or index % progress_every == 0 or index == total:
            print(f"[{split}] progress {index}/{total} | latest={target}", flush=True)

    manifest_path = output_dir / f"modelnet40_{split}.txt"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for entry in manifest_entries:
            handle.write(f"{entry}\n")

    return len(manifest_entries), summarize_stats(stats_list)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a mildly corrupted ModelNet40 dataset from OFF or h5 input."
    )
    parser.add_argument("--modelnet40-root", default="data/ModelNet40")
    parser.add_argument(
        "--input-format",
        default="auto",
        choices=["auto", "off", "h5"],
        help="How to read ModelNet40.",
    )
    parser.add_argument("--split", default="both", choices=["train", "test", "both"])
    parser.add_argument(
        "--output-root",
        default="data/ModelNet40_mild",
        help="Root folder for processed data.",
    )
    parser.add_argument(
        "--output-format",
        default="modelnet",
        choices=["modelnet", "h5"],
        help="Output as ModelNet-style directories or legacy h5 files.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=2048,
        help="Point count used only when reading OFF meshes.",
    )
    parser.add_argument(
        "--sample-method",
        default="surface",
        choices=["surface", "vertex"],
        help="How to sample points from OFF meshes.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-drop-min", type=float, default=0.03)
    parser.add_argument("--random-drop-max", type=float, default=0.08)
    parser.add_argument("--local-drop-min", type=float, default=0.05)
    parser.add_argument("--local-drop-max", type=float, default=0.12)
    parser.add_argument(
        "--min-keep-ratio",
        type=float,
        default=0.82,
        help="Guarantee at least this many original points remain before refill.",
    )
    parser.add_argument(
        "--normalize-before-chunk",
        action="store_true",
        help="Normalize each sample before computing local chunk distance.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    args = parser.parse_args()

    input_root = Path(args.modelnet40_root)
    input_format = infer_input_format(input_root) if args.input_format == "auto" else args.input_format
    output_root = Path(args.output_root)
    output_dir = output_root / "modelnet40_ply_hdf5_2048" if args.output_format == "h5" else output_root
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_format == "off":
        labels, train_samples, test_samples = build_modelnet40_splits(input_root)
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        split_to_samples: dict[str, list[dict[str, object]]] = {
            "train": [
                {
                    "path": sample["path"],
                    "sample_path": sample["path"],
                    "label_name": sample["label"],
                    "label_idx": label_to_idx[sample["label"]],
                }
                for sample in train_samples
            ],
            "test": [
                {
                    "path": sample["path"],
                    "sample_path": sample["path"],
                    "label_name": sample["label"],
                    "label_idx": label_to_idx[sample["label"]],
                }
                for sample in test_samples
            ],
        }
    else:
        train_labels, train_samples = load_h5_split_samples(input_root, "train")
        test_labels, test_samples = load_h5_split_samples(input_root, "test")
        if train_labels != test_labels:
            raise ValueError("Train and test shape_names do not match for the input h5 dataset.")
        labels = train_labels
        split_to_samples = {"train": train_samples, "test": test_samples}

    write_output_metadata(output_dir, labels)
    requested_splits = ["train", "test"] if args.split == "both" else [args.split]

    for split_index, split in enumerate(requested_splits):
        processed_data, labels_array, sample_paths, label_names, summary = process_samples(
            samples=split_to_samples[split],
            seed=args.seed + split_index,
            random_drop_min=args.random_drop_min,
            random_drop_max=args.random_drop_max,
            local_drop_min=args.local_drop_min,
            local_drop_max=args.local_drop_max,
            min_keep_ratio=args.min_keep_ratio,
            normalize_before_chunk=args.normalize_before_chunk,
            off_num_points=args.num_points,
            sample_method=args.sample_method,
        )

        if args.output_format == "h5":
            h5_filename = "ply_data_train0.h5" if split == "train" else "ply_data_test0.h5"
            output_h5 = output_dir / h5_filename
            if output_h5.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Output file already exists: {output_h5}. Use --overwrite to replace it."
                )

            string_dtype = h5py.string_dtype(encoding="utf-8")
            with h5py.File(output_h5, "w") as handle:
                handle.create_dataset("data", data=processed_data, compression="gzip")
                handle.create_dataset("label", data=labels_array, compression="gzip")
                handle.create_dataset(
                    "sample_path",
                    data=np.asarray(sample_paths, dtype=object),
                    dtype=string_dtype,
                )
                handle.create_dataset(
                    "label_name",
                    data=np.asarray(label_names, dtype=object),
                    dtype=string_dtype,
                )

            write_split_file(output_dir, split, h5_filename)
            output_summary = str(output_h5)
        else:
            print(f"[{split}] input_format={input_format} | output_format={args.output_format}", flush=True)
            print(f"[{split}] processing {len(split_to_samples[split])} samples into {output_dir}", flush=True)
            written_count, summary = process_modelnet_directory_split(
                output_dir=output_dir,
                split=split,
                labels=labels,
                samples=split_to_samples[split],
                seed=args.seed + split_index,
                random_drop_min=args.random_drop_min,
                random_drop_max=args.random_drop_max,
                local_drop_min=args.local_drop_min,
                local_drop_max=args.local_drop_max,
                min_keep_ratio=args.min_keep_ratio,
                normalize_before_chunk=args.normalize_before_chunk,
                off_num_points=args.num_points,
                sample_method=args.sample_method,
                overwrite=args.overwrite,
            )
            output_summary = f"{output_dir} ({written_count} files)"
            processed_data = None

        print(f"[{split}] output: {output_summary}")
        if processed_data is not None:
            print(f"[{split}] samples={len(processed_data)} | points_per_sample={processed_data.shape[1]}")
        else:
            print(f"[{split}] samples={len(split_to_samples[split])} | points_per_sample={args.num_points}")
        print(
            f"[{split}] mean_random_drop={summary['mean_random_drop_ratio']:.4f} "
            f"mean_local_drop={summary['mean_local_drop_ratio']:.4f} "
            f"mean_keep_before_refill={summary['mean_kept_ratio_before_refill']:.4f} "
            f"mean_final_unique={summary['mean_final_unique_ratio']:.4f}"
        )
        print(
            f"[{split}] random_mode_samples={int(summary['random_mode_count'])} "
            f"local_mode_samples={int(summary['local_mode_count'])}"
        )

    print(f"shape_names: {output_dir / 'shape_names.txt'}")
    if args.output_format == "h5":
        print(f"train_files : {output_dir / 'train_files.txt'}")
        print(f"test_files  : {output_dir / 'test_files.txt'}")
    else:
        print(f"train_split : {output_dir / 'modelnet40_train.txt'}")
        print(f"test_split  : {output_dir / 'modelnet40_test.txt'}")


if __name__ == "__main__":
    main()
