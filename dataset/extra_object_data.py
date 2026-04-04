import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSDF.dataset.scanobjectnn_data import SCANOBJECTNN_LABELS, ScanObjectNNDataset

try:
    import open3d as o3d
except Exception:
    o3d = None

try:
    import h5py
except Exception:
    h5py = None


POINT_FILE_EXTS = {".pcd", ".ply", ".npy", ".npz", ".txt", ".pts", ".xyz"}


def normalize_points(points):
    points = points.astype(np.float32)
    centroid = points.mean(axis=0, keepdims=True)
    points = points - centroid
    scale = np.linalg.norm(points, axis=1).max()
    if scale > 0:
        points = points / scale
    return points


def maybe_augment(points, rng):
    theta = rng.uniform(0.0, 2.0 * np.pi)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation = np.array(
        [[cos_theta, 0.0, sin_theta], [0.0, 1.0, 0.0], [-sin_theta, 0.0, cos_theta]],
        dtype=np.float32,
    )
    points = points @ rotation.T
    scale = rng.uniform(0.9, 1.1)
    points = points * scale
    translation = rng.uniform(-0.1, 0.1, size=(1, 3)).astype(np.float32)
    points = points + translation
    jitter = rng.normal(0.0, 0.01, size=points.shape).astype(np.float32)
    points = points + np.clip(jitter, -0.02, 0.02)

    dropout_ratio = rng.uniform(0.0, 0.15)
    if len(points) > 0 and dropout_ratio > 0:
        drop_mask = rng.random(len(points)) < dropout_ratio
        if np.any(drop_mask):
            points[drop_mask] = points[0]
    return points


def sample_points(points, num_points, rng):
    if len(points) >= num_points:
        indices = rng.choice(len(points), num_points, replace=False)
    else:
        indices = rng.choice(len(points), num_points, replace=True)
    return points[indices]


def extra_object_root_exists(root):
    root = Path(root)
    return root.exists() and any(path.is_dir() for path in root.iterdir())


def modelnet40_root_exists(root):
    root = Path(root)
    h5_candidate = root / "modelnet40_ply_hdf5_2048" if root.name != "modelnet40_ply_hdf5_2048" else root
    off_candidate = root / "ModelNet40" if root.name != "ModelNet40" else root
    return (
        (h5_candidate / "train_files.txt").exists() and (h5_candidate / "shape_names.txt").exists()
    ) or off_candidate.exists()


def discover_extra_object_labels(root):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def build_merged_labels(extra_root):
    labels = list(SCANOBJECTNN_LABELS)
    for label in discover_extra_object_labels(extra_root):
        if label not in labels:
            labels.append(label)
    return labels


def build_modelnet40_merged_labels(modelnet_labels, extra_root):
    labels = list(modelnet_labels)
    for label in discover_extra_object_labels(extra_root):
        if label not in labels:
            labels.append(label)
    return labels


def load_point_file(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        key = "points" if "points" in data else list(data.keys())[0]
        points = data[key]
    elif suffix in {".txt", ".pts", ".xyz"}:
        points = np.loadtxt(path)
    elif suffix in {".pcd", ".ply"}:
        if o3d is None:
            raise RuntimeError("Open3D is required to load .pcd/.ply extra object files.")
        cloud = o3d.io.read_point_cloud(str(path))
        if cloud.is_empty():
            raise ValueError(f"Point cloud is empty: {path}")
        points = np.asarray(cloud.points, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported extra object file format: {path}")

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected Nx3+ points in {path}, got shape {points.shape}")
    return points[:, :3]


class RemappedClassificationDataset(Dataset):
    def __init__(self, base_dataset, base_labels, merged_labels):
        self.base_dataset = base_dataset
        self.base_labels = list(base_labels)
        self.merged_labels = list(merged_labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.merged_labels)}
        self.label_map = np.asarray([self.label_to_idx[label] for label in self.base_labels], dtype=np.int64)
        self.labels = self.label_map[np.asarray(base_dataset.labels, dtype=np.int64)]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        points, label = self.base_dataset[idx]
        return points, int(self.label_map[int(label)])


class ExtraObjectDataset(Dataset):
    def __init__(self, root, labels, num_points=1024, split="train", augment=False, seed=0):
        self.root = Path(root)
        self.labels_list = list(labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels_list)}
        self.num_points = num_points
        self.split = split
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.samples = []

        for label_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            label = label_dir.name
            if label not in self.label_to_idx:
                continue

            split_dir = label_dir / split
            if split_dir.exists():
                search_roots = [split_dir]
            elif split == "train":
                search_roots = [label_dir]
            else:
                search_roots = []

            for search_root in search_roots:
                for path in sorted(search_root.rglob("*")):
                    if not path.is_file() or path.suffix.lower() not in POINT_FILE_EXTS:
                        continue
                    if split == "train" and "test" in path.parts:
                        continue
                    self.samples.append({"path": str(path), "label_idx": self.label_to_idx[label]})

        self.labels = np.asarray([sample["label_idx"] for sample in self.samples], dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        points = load_point_file(sample["path"])
        points = normalize_points(points)
        points = sample_points(points, self.num_points, self.rng)
        if self.augment:
            points = maybe_augment(points, self.rng)
        return torch.from_numpy(points.T.astype(np.float32)), int(sample["label_idx"])


class ConcatClassificationDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = [dataset for dataset in datasets if dataset is not None and len(dataset) > 0]
        self.cumulative_sizes = []
        total = 0
        labels = []
        for dataset in self.datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)
            if hasattr(dataset, "labels"):
                labels.append(np.asarray(dataset.labels, dtype=np.int64))
        self.labels = np.concatenate(labels, axis=0) if labels else np.zeros(0, dtype=np.int64)

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        for dataset_idx, cumulative_size in enumerate(self.cumulative_sizes):
            if idx < cumulative_size:
                prev_cumulative = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
                return self.datasets[dataset_idx][idx - prev_cumulative]
        raise IndexError(idx)


def load_off_mesh(path):
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
            n = parts[0]
            face = parts[1 : 1 + n]
            if len(face) >= 3:
                for i in range(1, len(face) - 1):
                    faces.append([face[0], face[i], face[i + 1]])

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"OFF file has no valid vertices or faces: {path}")

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

    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"OFF file has no valid vertices or faces after filtering: {path}")
    return vertices, faces


def sample_points_from_mesh(vertices, faces, num_points, rng):
    safe_vertices = vertices[np.isfinite(vertices).all(axis=1)]
    if len(safe_vertices) == 0:
        raise ValueError("Mesh contains no finite vertices for sampling.")

    safe_vertices64 = safe_vertices.astype(np.float64, copy=False)
    scale = np.abs(safe_vertices64).max()
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    if scale > 1e6:
        safe_vertices64 = safe_vertices64 / scale

    triangles = safe_vertices64[faces]
    vec1 = triangles[:, 1] - triangles[:, 0]
    vec2 = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(vec1, vec2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    finite_area_mask = np.isfinite(areas) & (areas > 0)

    if not np.any(finite_area_mask):
        return sample_points(safe_vertices.astype(np.float32), num_points, rng).astype(np.float32)

    triangles = triangles[finite_area_mask]
    areas = areas[finite_area_mask]
    area_sum = areas.sum()
    if not np.isfinite(area_sum) or area_sum <= 0:
        return sample_points(safe_vertices.astype(np.float32), num_points, rng).astype(np.float32)

    probs = areas / area_sum
    probs = probs / probs.sum()
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


class ModelNet40H5Dataset(Dataset):
    def __init__(self, root, split, num_points=1024, augment=False, seed=0):
        if h5py is None:
            raise RuntimeError("h5py is required to read ModelNet40 HDF5 data.")

        self.root = Path(root)
        self.split = split
        self.num_points = num_points
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        if self.root.name != "modelnet40_ply_hdf5_2048" and (self.root / "modelnet40_ply_hdf5_2048").exists():
            self.root = self.root / "modelnet40_ply_hdf5_2048"

        self.labels = self._load_labels()
        self.samples = self._load_split_samples(split)

    def _load_labels(self):
        labels_path = self.root / "shape_names.txt"
        if not labels_path.exists():
            raise FileNotFoundError(f"Could not find {labels_path}")
        with open(labels_path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    def _load_split_samples(self, split):
        filelist_name = "train_files.txt" if split == "train" else "test_files.txt"
        filelist_path = self.root / filelist_name
        if not filelist_path.exists():
            raise FileNotFoundError(f"Could not find {filelist_path}")

        samples = []
        with open(filelist_path, "r", encoding="utf-8") as handle:
            relative_paths = [line.strip() for line in handle if line.strip()]

        for relpath in relative_paths:
            h5_name = Path(relpath).name
            h5_path = self.root / h5_name
            if not h5_path.exists():
                h5_path = self.root / relpath
            if not h5_path.exists():
                raise FileNotFoundError(f"Could not find H5 file: {h5_path}")

            with h5py.File(h5_path, "r") as data:
                points = np.asarray(data["data"], dtype=np.float32)[:, :, :3]
                labels = np.asarray(data["label"]).reshape(-1).astype(np.int64)
            for idx in range(len(labels)):
                samples.append({"points": points[idx], "label_idx": int(labels[idx])})
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        points = normalize_points(sample["points"])
        points = sample_points(points, self.num_points, self.rng)
        if self.augment:
            points = maybe_augment(points, self.rng)
        return torch.from_numpy(points.T.astype(np.float32)), sample["label_idx"]


class ModelNet40OffDataset(Dataset):
    def __init__(self, root, split, num_points=1024, augment=False, seed=0):
        self.root = Path(root)
        self.split = split
        self.num_points = num_points
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        nested_root = self.root / "ModelNet40"
        if nested_root.exists() and any(path.is_dir() for path in nested_root.iterdir()):
            self.root = nested_root

        if not self.root.exists():
            raise FileNotFoundError(f"Could not find ModelNet40 directory: {self.root}")

        self.labels = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        if not self.labels:
            raise RuntimeError(f"No class directories found under {self.root}.")

        self.samples = []
        for label_idx, label in enumerate(self.labels):
            split_dir = self.root / label / split
            if not split_dir.exists():
                continue
            for path in sorted(split_dir.glob("*.off")):
                self.samples.append({"path": path, "label_idx": label_idx})

        if not self.samples:
            raise RuntimeError(f"No OFF files found for split '{split}' under {self.root}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        vertices, faces = load_off_mesh(sample["path"])
        try:
            points = sample_points_from_mesh(vertices, faces, self.num_points, self.rng)
        except ValueError:
            safe_vertices = vertices[np.isfinite(vertices).all(axis=1)]
            if len(safe_vertices) == 0:
                raise
            points = sample_points(safe_vertices.astype(np.float32), self.num_points, self.rng).astype(np.float32)
        points = normalize_points(points)
        if self.augment:
            points = maybe_augment(points, self.rng)
        return torch.from_numpy(points.T.astype(np.float32)), sample["label_idx"]


def get_modelnet40_with_extra_dataloaders(
    modelnet40_root,
    extra_object_root,
    batch_size=32,
    num_points=1024,
    workers=4,
    seed=0,
    include_extra=True,
):
    modelnet_root = Path(modelnet40_root)
    h5_ready = (
        (modelnet_root / "modelnet40_ply_hdf5_2048" / "train_files.txt").exists()
        or (modelnet_root / "train_files.txt").exists()
    )

    if h5_ready:
        base_train = ModelNet40H5Dataset(
            root=modelnet40_root,
            split="train",
            num_points=num_points,
            augment=True,
            seed=seed,
        )
        base_test = ModelNet40H5Dataset(
            root=modelnet40_root,
            split="test",
            num_points=num_points,
            augment=False,
            seed=seed + 1,
        )
    else:
        base_train = ModelNet40OffDataset(
            root=modelnet40_root,
            split="train",
            num_points=num_points,
            augment=True,
            seed=seed,
        )
        base_test = ModelNet40OffDataset(
            root=modelnet40_root,
            split="test",
            num_points=num_points,
            augment=False,
            seed=seed + 1,
        )

    labels = build_modelnet40_merged_labels(base_train.labels, extra_object_root if include_extra else None)
    train_parts = [RemappedClassificationDataset(base_train, base_train.labels, labels)]
    test_parts = [RemappedClassificationDataset(base_test, base_test.labels, labels)]

    if include_extra and extra_object_root_exists(extra_object_root):
        extra_train = ExtraObjectDataset(
            root=extra_object_root,
            labels=labels,
            num_points=num_points,
            split="train",
            augment=True,
            seed=seed + 2,
        )
        if len(extra_train) > 0:
            train_parts.append(extra_train)

        extra_test = ExtraObjectDataset(
            root=extra_object_root,
            labels=labels,
            num_points=num_points,
            split="test",
            augment=False,
            seed=seed + 3,
        )
        if len(extra_test) > 0:
            test_parts.append(extra_test)

    train_dataset = ConcatClassificationDataset(train_parts)
    test_dataset = ConcatClassificationDataset(test_parts)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        drop_last=False,
    )
    return labels, train_dataset, test_dataset, train_loader, test_loader


def get_scanobjectnn_with_extra_dataloaders(
    scanobjectnn_root,
    extra_object_root,
    variant="pb_t50_rs",
    batch_size=32,
    num_points=1024,
    workers=4,
    use_background=True,
    seed=0,
    include_extra=True,
):
    labels = build_merged_labels(extra_object_root if include_extra else None)

    base_train = ScanObjectNNDataset(
        root=scanobjectnn_root,
        split="train",
        variant=variant,
        num_points=num_points,
        use_background=use_background,
        normalize=True,
        augment=True,
        seed=seed,
    )
    base_test = ScanObjectNNDataset(
        root=scanobjectnn_root,
        split="test",
        variant=variant,
        num_points=num_points,
        use_background=use_background,
        normalize=True,
        augment=False,
        seed=seed + 1,
    )

    train_parts = [RemappedClassificationDataset(base_train, SCANOBJECTNN_LABELS, labels)]
    test_parts = [RemappedClassificationDataset(base_test, SCANOBJECTNN_LABELS, labels)]

    if include_extra and extra_object_root_exists(extra_object_root):
        extra_train = ExtraObjectDataset(
            root=extra_object_root,
            labels=labels,
            num_points=num_points,
            split="train",
            augment=True,
            seed=seed + 2,
        )
        if len(extra_train) > 0:
            train_parts.append(extra_train)

        extra_test = ExtraObjectDataset(
            root=extra_object_root,
            labels=labels,
            num_points=num_points,
            split="test",
            augment=False,
            seed=seed + 3,
        )
        if len(extra_test) > 0:
            test_parts.append(extra_test)

    train_dataset = ConcatClassificationDataset(train_parts)
    test_dataset = ConcatClassificationDataset(test_parts)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        drop_last=False,
    )
    return labels, train_dataset, test_dataset, train_loader, test_loader
