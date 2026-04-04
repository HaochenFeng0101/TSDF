import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import trimesh
except Exception:
    trimesh = None

try:
    import open3d as o3d
except Exception:
    o3d = None

from TSDF.dataset.scanobjectnn_data import maybe_augment, normalize_points


POINT_EXTENSIONS = {".off", ".npy", ".npz", ".txt", ".pts", ".xyz", ".pcd", ".ply"}


def _resolve_root(root):
    root = Path(root)
    candidates = [root, root / "ModelNet40", root / "modelnet40"]
    for candidate in candidates:
        if candidate.exists() and _looks_like_modelnet_root(candidate):
            return candidate

    for candidate in candidates:
        if not candidate.exists():
            continue
        for subdir in sorted(path for path in candidate.rglob("*") if path.is_dir()):
            if _looks_like_modelnet_root(subdir):
                return subdir
    raise FileNotFoundError(f"Could not find ModelNet40 root under {root}")


def _looks_like_modelnet_root(path):
    path = Path(path)
    if not path.exists() or not path.is_dir():
        return False

    split_files = [
        path / "modelnet40_train.txt",
        path / "modelnet40_test.txt",
        path / "train_files.txt",
        path / "test_files.txt",
    ]
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
        if any(child.suffix.lower() in POINT_EXTENSIONS for child in class_dir.glob("*")):
            return True
    return False


def _read_shape_names(root):
    candidates = [
        root / "modelnet40_shape_names.txt",
        root / "shape_names.txt",
        root / "labels.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as handle:
                labels = [line.strip() for line in handle if line.strip()]
            if labels:
                return labels

    labels = sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if not labels:
        raise ValueError(f"No class directories found under {root}")
    return labels


def _scan_split_dirs(root, split, labels):
    samples = []
    for label in labels:
        split_dir = root / label / split
        if not split_dir.exists():
            label_dir = root / label
            if label_dir.exists():
                for path in sorted(label_dir.iterdir()):
                    if path.is_file() and path.suffix.lower() in POINT_EXTENSIONS and split in path.stem:
                        samples.append({"path": str(path), "label": label})
            continue
        for path in sorted(split_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in POINT_EXTENSIONS:
                samples.append({"path": str(path), "label": label})
    return samples


def _find_sample_path(root, label, stem):
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


def _read_split_list(root, split, labels):
    split_candidates = [
        root / f"modelnet40_{split}.txt",
        root / f"{split}_files.txt",
    ]
    split_file = None
    for candidate in split_candidates:
        if candidate.exists():
            split_file = candidate
            break
    if split_file is None:
        return None

    label_set = set(labels)
    samples = []
    with open(split_file, "r", encoding="utf-8") as handle:
        for line in handle:
            item = line.strip().replace("\\", "/")
            if not item:
                continue
            stem = Path(item).stem
            matched_label = None
            for label in label_set:
                prefix = f"{label}_"
                if stem.startswith(prefix):
                    matched_label = label
                    break
            if matched_label is None:
                continue
            actual_stem = stem[len(matched_label) + 1 :]
            samples.append(
                {
                    "path": str(_find_sample_path(root, matched_label, actual_stem)),
                    "label": matched_label,
                }
            )
    return samples


def build_modelnet40_splits(root):
    root = _resolve_root(root)
    labels = _read_shape_names(root)

    train_samples = _read_split_list(root, "train", labels)
    test_samples = _read_split_list(root, "test", labels)

    if train_samples is None or test_samples is None:
        train_samples = _scan_split_dirs(root, "train", labels)
        test_samples = _scan_split_dirs(root, "test", labels)

    if not train_samples or not test_samples:
        raise ValueError(
            f"Could not build ModelNet40 train/test splits under {root}. "
            "Expected either modelnet40_train.txt/modelnet40_test.txt or class/train|test folders."
        )
    return labels, train_samples, test_samples


def load_point_cloud_file(path):
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
            raise RuntimeError(
                f"open3d is required to read {suffix} files. Install it before training."
            )
        point_cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(point_cloud.points)
    else:
        raise ValueError(f"Unsupported point file format: {path}")

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected Nx3+ points in {path}, got shape {points.shape}")
    return np.asarray(points[:, :3], dtype=np.float32)


def load_off_points(path, num_points, rng, sample_method="surface"):
    if trimesh is None:
        raise RuntimeError(
            "trimesh is required to load ModelNet40 .off files. Install it before training."
        )

    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = [geom for geom in mesh.geometry.values()]
        if not geometries:
            raise ValueError(f"Empty scene in {path}")
        mesh = trimesh.util.concatenate(tuple(geometries))

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"Expected non-empty Nx3 vertices in {path}")

    if sample_method == "surface" and hasattr(mesh, "faces") and len(mesh.faces) > 0:
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        return points.astype(np.float32)

    replace = len(vertices) < num_points
    indices = rng.choice(len(vertices), num_points, replace=replace)
    return vertices[indices].astype(np.float32)


class ModelNet40Dataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        num_points=1024,
        normalize=True,
        augment=False,
        seed=0,
        sample_method="surface",
    ):
        self.root = _resolve_root(root)
        self.split = split
        self.num_points = num_points
        self.normalize = normalize
        self.augment = augment
        self.sample_method = sample_method
        self.rng = np.random.default_rng(seed)

        labels, train_samples, test_samples = build_modelnet40_splits(self.root)
        self.labels = labels
        self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        self.samples = train_samples if split == "train" else test_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_path = Path(sample["path"])
        if sample_path.suffix.lower() == ".off":
            points = load_off_points(
                sample_path,
                num_points=self.num_points,
                rng=self.rng,
                sample_method=self.sample_method,
            )
        else:
            points = load_point_cloud_file(sample_path)
            if len(points) >= self.num_points:
                choice = self.rng.choice(len(points), self.num_points, replace=False)
            else:
                choice = self.rng.choice(len(points), self.num_points, replace=True)
            points = points[choice].astype(np.float32)
        if self.normalize:
            points = normalize_points(points)
        if self.augment:
            points = maybe_augment(points, self.rng)
        tensor = torch.from_numpy(points.T.astype(np.float32))
        label = self.label_to_idx[sample["label"]]
        return tensor, label


def get_modelnet40_dataloaders(
    root,
    batch_size=32,
    num_points=1024,
    workers=4,
    seed=0,
    sample_method="surface",
):
    train_dataset = ModelNet40Dataset(
        root=root,
        split="train",
        num_points=num_points,
        normalize=True,
        augment=True,
        seed=seed,
        sample_method=sample_method,
    )
    test_dataset = ModelNet40Dataset(
        root=root,
        split="test",
        num_points=num_points,
        normalize=True,
        augment=False,
        seed=seed + 1,
        sample_method=sample_method,
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
