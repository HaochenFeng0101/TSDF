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
