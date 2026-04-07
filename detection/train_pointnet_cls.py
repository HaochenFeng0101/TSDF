import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSDF.dataset.extra_object_data import load_point_file
from TSDF.dataset.scanobjectnn_data import maybe_augment, normalize_points

try:
    import h5py
except Exception:
    h5py = None


POINT_FILE_EXTS = {".pcd", ".ply", ".npy", ".npz", ".txt", ".pts", ".xyz"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_points(points, num_points, rng):
    if len(points) >= num_points:
        indices = rng.choice(len(points), num_points, replace=False)
    else:
        indices = rng.choice(len(points), num_points, replace=True)
    return points[indices]


def build_dir_splits(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Could not find dataset root: {root}")

    labels = sorted(path.name for path in root.iterdir() if path.is_dir())
    train_samples = []
    test_samples = []

    for label in labels:
        label_root = root / label
        for split_name, target in (("train", train_samples), ("test", test_samples)):
            split_dir = label_root / split_name
            if not split_dir.exists():
                continue
            for path in sorted(split_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in POINT_FILE_EXTS:
                    target.append({"path": str(path), "label": label})

    if not labels:
        raise ValueError(f"No class directories found under {root}")
    if not train_samples:
        raise ValueError(f"No training samples found under {root}")
    if not test_samples:
        raise ValueError(f"No test samples found under {root}")

    return labels, train_samples, test_samples


def load_h5_samples(path):
    if h5py is None:
        raise RuntimeError("h5py is required to read H5 classification files.")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find H5 file: {path}")

    with h5py.File(path, "r") as handle:
        data = np.asarray(handle["data"], dtype=np.float32)
        labels = np.asarray(handle["label"]).reshape(-1).astype(np.int64)

    samples = []
    for points, label_idx in zip(data, labels):
        samples.append({"points": np.asarray(points[:, :3], dtype=np.float32), "label_idx": int(label_idx)})
    return samples


class _BaseClassificationDataset(Dataset):
    def __init__(self, samples, num_points, labels, augment=False, seed=0):
        self.samples = list(samples)
        self.num_points = int(num_points)
        self.labels = list(labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.augment = augment
        self.seed = seed

    def __len__(self):
        return len(self.samples)

    def _load_points_and_label(self, sample):
        if "points" in sample:
            points = np.asarray(sample["points"], dtype=np.float32)
        else:
            points = load_point_file(sample["path"])

        if "label_idx" in sample:
            label_idx = int(sample["label_idx"])
        else:
            label_idx = self.label_to_idx[sample["label"]]

        return points[:, :3], label_idx

    def __getitem__(self, idx):
        sample = self.samples[idx]
        rng = np.random.default_rng(self.seed + idx)
        points, label_idx = self._load_points_and_label(sample)
        points = normalize_points(points)
        points = _sample_points(points, self.num_points, rng)
        if self.augment:
            points = maybe_augment(points, rng)
        return torch.from_numpy(points.T.astype(np.float32)), int(label_idx)


class PointCloudClassificationDataset(_BaseClassificationDataset):
    pass


class H5ClassificationDataset(_BaseClassificationDataset):
    pass
