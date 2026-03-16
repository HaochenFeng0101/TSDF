import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import h5py
except Exception:
    h5py = None


SCANOBJECTNN_LABELS = [
    "bag",
    "bin",
    "box",
    "cabinet",
    "chair",
    "desk",
    "display",
    "door",
    "shelf",
    "table",
    "bed",
    "pillow",
    "sink",
    "sofa",
    "toilet",
]

VARIANT_TO_FILES = {
    "obj_bg": (
        "training_objectdataset.h5",
        "test_objectdataset.h5",
    ),
    "obj_only": (
        "training_objectdataset.h5",
        "test_objectdataset.h5",
    ),
    "pb_t25": (
        "training_objectdataset_augmented25_norot.h5",
        "test_objectdataset_augmented25_norot.h5",
    ),
    "pb_t25_r": (
        "training_objectdataset_augmented25rot.h5",
        "test_objectdataset_augmented25rot.h5",
    ),
    "pb_t50_r": (
        "training_objectdataset_augmentedrot.h5",
        "test_objectdataset_augmentedrot.h5",
    ),
    "pb_t50_rs": (
        "training_objectdataset_augmentedrot_scale75.h5",
        "test_objectdataset_augmentedrot_scale75.h5",
    ),
}


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
    jitter = rng.normal(0.0, 0.01, size=points.shape).astype(np.float32)
    return points + np.clip(jitter, -0.02, 0.02)


def _resolve_split_dir(root, split_name):
    root = Path(root)
    candidates = [root / split_name, root / "h5_files" / split_name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find split directory '{split_name}' under {root}"
    )


def _read_h5_file(path):
    if h5py is None:
        raise RuntimeError(
            "h5py is not installed. Install it with `conda install -n MonoGS h5py`."
        )
    with h5py.File(path, "r") as handle:
        data = np.asarray(handle["data"], dtype=np.float32)
        labels = np.asarray(handle["label"]).reshape(-1).astype(np.int64)
        mask = np.asarray(handle["mask"], dtype=np.float32) if "mask" in handle else None
    return data, labels, mask


class ScanObjectNNDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        variant="pb_t50_rs",
        num_points=1024,
        use_background=True,
        normalize=True,
        augment=False,
        seed=0,
    ):
        self.root = Path(root)
        self.split = split
        self.variant = variant.lower()
        self.num_points = num_points
        self.use_background = use_background
        self.normalize = normalize
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        if self.variant not in VARIANT_TO_FILES:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from {sorted(VARIANT_TO_FILES)}"
            )

        split_dir_name = "main_split" if use_background else "main_split_nobg"
        split_dir = _resolve_split_dir(self.root, split_dir_name)
        file_idx = 0 if split == "train" else 1
        filename = VARIANT_TO_FILES[self.variant][file_idx]
        data_path = split_dir / filename
        if not data_path.exists():
            raise FileNotFoundError(f"Could not find {data_path}")

        self.data, self.labels, self.mask = _read_h5_file(data_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        points = self.data[idx][:, :3].copy()
        if self.normalize:
            points = normalize_points(points)

        if len(points) >= self.num_points:
            choice = self.rng.choice(len(points), self.num_points, replace=False)
        else:
            choice = self.rng.choice(len(points), self.num_points, replace=True)
        points = points[choice]

        if self.augment:
            points = maybe_augment(points, self.rng)

        points = torch.from_numpy(points.T.astype(np.float32))
        label = int(self.labels[idx])
        return points, label


def get_scanobjectnn_dataloaders(
    root,
    variant="pb_t50_rs",
    batch_size=32,
    num_points=1024,
    workers=4,
    use_background=True,
    seed=0,
):
    train_dataset = ScanObjectNNDataset(
        root=root,
        split="train",
        variant=variant,
        num_points=num_points,
        use_background=use_background,
        normalize=True,
        augment=True,
        seed=seed,
    )
    test_dataset = ScanObjectNNDataset(
        root=root,
        split="test",
        variant=variant,
        num_points=num_points,
        use_background=use_background,
        normalize=True,
        augment=False,
        seed=seed + 1,
    )

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
    return train_dataset, test_dataset, train_loader, test_loader
