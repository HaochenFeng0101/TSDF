import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset



# python3 detection/pointnet2/train.py \
#   --dataset-type modelnet40 \
#   --modelnet40-root data/ModelNet40 \
#   --epochs 150 \
#   --batch-size 16 \
#   --num-points 1024

'''

python3 detection/pointnet2/train.py \
  --dataset-type modelnet40 \
  --modelnet40-root data/ModelNet40 \
  --extra-object-root data/extra_object \
  --epochs 150 \
  --batch-size 16 \
  --num-points 1024
'''

REPO_ROOT = Path(__file__).resolve().parents[3]
TSDF_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSDF.dataset.extra_object_data import (
    ModelNet40H5Dataset as ExtraModelNet40H5Dataset,
    ModelNet40OffDataset as ExtraModelNet40OffDataset,
    get_modelnet40_with_extra_dataloaders,
    get_scanobjectnn_with_extra_dataloaders,
    modelnet40_root_exists,
)
from TSDF.dataset.scanobjectnn_data import SCANOBJECTNN_LABELS, ScanObjectNNDataset
from TSDF.detection.pointnet2.pointnet2 import PointNet2ClsSSG

try:
    import h5py
except Exception:
    h5py = None

try:
    import open3d as o3d
except Exception:
    o3d = None

try:
    import wandb
except Exception:
    wandb = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_points(points, num_points, rng):
    if len(points) >= num_points:
        indices = rng.choice(len(points), num_points, replace=False)
    else:
        indices = rng.choice(len(points), num_points, replace=True)
    return points[indices]


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
    return points


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

    valid_face_mask = (
        (faces >= 0).all(axis=1)
        & (faces < len(vertices)).all(axis=1)
    )
    faces = faces[valid_face_mask]

    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"OFF file has no valid vertices or faces after filtering: {path}")
    return vertices, faces


POINT_EXTENSIONS = {".off", ".npy", ".npz", ".txt", ".pts", ".xyz", ".pcd", ".ply"}


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
            raise RuntimeError(f"open3d is required to read {suffix} files. Install it before training.")
        point_cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(point_cloud.points)
    else:
        raise ValueError(f"Unsupported point file format: {path}")

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected Nx3+ points in {path}, got shape {points.shape}")
    return points[:, :3]


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
            for path in sorted(split_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in POINT_EXTENSIONS:
                    self.samples.append({"path": path, "label_idx": label_idx})

        if not self.samples:
            raise RuntimeError(f"No ModelNet40 files found for split '{split}' under {self.root}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_path = Path(sample["path"])
        if sample_path.suffix.lower() == ".off":
            vertices, faces = load_off_mesh(sample_path)
            try:
                points = sample_points_from_mesh(vertices, faces, self.num_points, self.rng)
            except ValueError:
                safe_vertices = vertices[np.isfinite(vertices).all(axis=1)]
                if len(safe_vertices) == 0:
                    raise
                points = sample_points(safe_vertices.astype(np.float32), self.num_points, self.rng).astype(np.float32)
        else:
            raw_points = load_point_cloud_file(sample_path)
            points = sample_points(raw_points.astype(np.float32), self.num_points, self.rng).astype(np.float32)
        points = normalize_points(points)
        if self.augment:
            points = maybe_augment(points, self.rng)
        return torch.from_numpy(points.T.astype(np.float32)), sample["label_idx"]


def scanobjectnn_is_ready(root):
    root = Path(root)
    return (
        root.exists()
        and (
            (root / "h5_files" / "main_split").exists()
            or (root / "main_split").exists()
        )
    )


def modelnet40_is_ready(root):
    root = Path(root)
    h5_candidate = root / "modelnet40_ply_hdf5_2048" if root.name != "modelnet40_ply_hdf5_2048" else root
    off_candidate = root / "ModelNet40" if root.name != "ModelNet40" else root
    return (
        (h5_candidate / "train_files.txt").exists() and (h5_candidate / "shape_names.txt").exists()
    ) or off_candidate.exists()


def resolve_dataset_type(args):
    if args.dataset_type != "auto":
        return args.dataset_type
    if scanobjectnn_is_ready(args.scanobjectnn_root):
        return "scanobjectnn"
    if modelnet40_is_ready(args.modelnet40_root):
        return "modelnet40"
    raise RuntimeError(
        "No usable dataset was found. Use an existing ScanObjectNN setup or run "
        "`python3 detection/pointnet2/download_modelnet40.py` to download ModelNet40."
    )


def evaluate(model, dataloader, device, class_weights=None, label_smoothing=0.0):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    iterator = dataloader
    if tqdm is not None:
        iterator = tqdm(dataloader, desc="val", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for points, labels in iterator:
            points = points.to(device)
            labels = labels.to(device)
            logits = model(points)
            loss = F.cross_entropy(
                logits,
                labels,
                weight=class_weights,
                label_smoothing=label_smoothing,
            )
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_seen += labels.size(0)
    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


def build_dataloader(dataset, batch_size, workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=False,
    )


def collect_label_indices(dataset):
    if isinstance(dataset, ConcatDataset):
        indices = []
        for subdataset in dataset.datasets:
            indices.extend(collect_label_indices(subdataset))
        return indices

    if isinstance(dataset, Subset):
        subset_labels = collect_label_indices(dataset.dataset)
        return [subset_labels[int(i)] for i in dataset.indices]

    if hasattr(dataset, "samples"):
        samples = getattr(dataset, "samples")
        if len(samples) > 0 and isinstance(samples[0], dict) and "label_idx" in samples[0]:
            return [int(sample["label_idx"]) for sample in samples]

    if hasattr(dataset, "labels"):
        labels = getattr(dataset, "labels")
        if len(labels) > 0:
            first = labels[0]
            if isinstance(first, (int, np.integer)) or hasattr(first, "item"):
                return [int(x) for x in labels]

    raise ValueError(f"Cannot infer label indices from dataset type: {type(dataset).__name__}")


def compute_class_weights(dataset, num_classes):
    raw_labels = np.asarray(collect_label_indices(dataset), dtype=np.int64)
    counts = np.bincount(raw_labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


def compute_sampled_processed_size(dataset_size, ratio_denominator):
    ratio_denominator = int(ratio_denominator)
    if ratio_denominator <= 0:
        raise ValueError("--mild-ratio-denominator must be a positive integer.")
    if dataset_size <= 0:
        return 0
    if ratio_denominator == 1:
        return dataset_size
    return max(1, dataset_size // ratio_denominator)


def sample_processed_dataset(dataset, target_size, seed):
    if target_size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:target_size].tolist()
    return Subset(dataset, indices)


def load_processed_scanobjectnn_dataset(args, labels):
    expected_labels = list(SCANOBJECTNN_LABELS)
    if list(labels[: len(expected_labels)]) != expected_labels:
        raise ValueError("Merged labels do not preserve the ScanObjectNN base label order.")
    return ScanObjectNNDataset(
        root=args.scanobjectnn_mild_root,
        split="train",
        variant=args.scanobjectnn_variant,
        num_points=args.num_points,
        use_background=not args.scanobjectnn_no_bg,
        normalize=True,
        augment=True,
        seed=args.seed + 11,
    )


def load_processed_modelnet40_dataset(args, labels):
    if not modelnet40_root_exists(args.modelnet40_mild_root):
        raise FileNotFoundError(
            f"Processed ModelNet40 root does not look valid: {Path(args.modelnet40_mild_root).resolve()}"
        )

    mild_root = Path(args.modelnet40_mild_root)
    h5_ready = (
        (mild_root / "modelnet40_ply_hdf5_2048" / "train_files.txt").exists()
        or (mild_root / "train_files.txt").exists()
    )

    if h5_ready:
        dataset = ExtraModelNet40H5Dataset(
            root=args.modelnet40_mild_root,
            split="train",
            num_points=args.num_points,
            augment=True,
            seed=args.seed + 11,
        )
        processed_format = "h5"
    else:
        dataset = ExtraModelNet40OffDataset(
            root=args.modelnet40_mild_root,
            split="train",
            num_points=args.num_points,
            augment=True,
            seed=args.seed + 11,
        )
        processed_format = "off"

    base_labels = list(dataset.labels)
    if list(labels[: len(base_labels)]) != base_labels:
        raise ValueError("Merged labels do not preserve the ModelNet40 base label order.")
    return dataset, processed_format


def main():
    parser = argparse.ArgumentParser(
        description="Train an original-style PointNet++ SSG classification model. Prefer ScanObjectNN and fall back to ModelNet40."
    )
    parser.add_argument(
        "--dataset-type",
        choices=["auto", "scanobjectnn", "modelnet40"],
        default="auto",
        help="auto prefers ScanObjectNN and falls back to ModelNet40.",
    )
    parser.add_argument(
        "--scanobjectnn-root",
        default=str(TSDF_ROOT / "data" / "ScanObjectNN"),
        help="ScanObjectNN root directory.",
    )
    parser.add_argument(
        "--scanobjectnn-variant",
        default="pb_t50_rs",
        help="ScanObjectNN variant.",
    )
    parser.add_argument(
        "--scanobjectnn-no-bg",
        action="store_true",
        help="Use the no-background split.",
    )
    parser.add_argument(
        "--extra-object-root",
        default=str(TSDF_ROOT / "data" / "extra_object"),
        help="Optional extra object directory. Class names are inferred from folder names and are included by default if the directory exists.",
    )
    parser.add_argument(
        "--no-extra-object-data",
        action="store_true",
        help="Disable loading extra object samples from --extra-object-root.",
    )
    parser.add_argument(
        "--modelnet40-root",
        default=str(TSDF_ROOT / "data" / "ModelNet40"),
        help="ModelNet40 root directory.",
    )
    parser.add_argument(
        "--use-processed-train-data",
        action="store_true",
        help="Append a sampled processed/mild train split to the original training set.",
    )
    parser.add_argument(
        "--scanobjectnn-mild-root",
        default=str(TSDF_ROOT / "data" / "ScanObjectNN_mild"),
        help="Processed ScanObjectNN root used when --use-processed-train-data is enabled.",
    )
    parser.add_argument(
        "--modelnet40-mild-root",
        default=str(TSDF_ROOT / "data" / "ModelNet40_mild"),
        help="Processed ModelNet40 root used when --use-processed-train-data is enabled.",
    )
    parser.add_argument(
        "--mild-ratio-denominator",
        type=int,
        default=5,
        help="Sample about 1/N of the processed train split when --use-processed-train-data is enabled.",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=str(TSDF_ROOT / "model" / "pointnet2"),
        help="Checkpoint output directory.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="TSDF-PointNet2")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_type = resolve_dataset_type(args)
    print(f"Using dataset_type={dataset_type}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_train_dataset_full = None
    processed_train_dataset = None
    processed_dataset_root = None
    processed_dataset_format = None

    if dataset_type == "scanobjectnn":
        labels, train_dataset, test_dataset, train_loader, test_loader = get_scanobjectnn_with_extra_dataloaders(
            scanobjectnn_root=args.scanobjectnn_root,
            extra_object_root=args.extra_object_root,
            variant=args.scanobjectnn_variant,
            batch_size=args.batch_size,
            num_points=args.num_points,
            workers=args.workers,
            use_background=not args.scanobjectnn_no_bg,
            seed=args.seed,
            include_extra=not args.no_extra_object_data,
        )
        print(
            f"scanobjectnn_extra_object_root="
            f"{args.extra_object_root if not args.no_extra_object_data else 'disabled'}"
        )
        if args.use_processed_train_data:
            processed_dataset_root = Path(args.scanobjectnn_mild_root).expanduser().resolve()
            processed_train_dataset_full = load_processed_scanobjectnn_dataset(args, labels)
    else:
        labels, train_dataset, test_dataset, train_loader, test_loader = get_modelnet40_with_extra_dataloaders(
            modelnet40_root=args.modelnet40_root,
            extra_object_root=args.extra_object_root,
            batch_size=args.batch_size,
            num_points=args.num_points,
            workers=args.workers,
            seed=args.seed,
            include_extra=not args.no_extra_object_data,
        )
        print(
            f"modelnet40_extra_object_root="
            f"{args.extra_object_root if not args.no_extra_object_data else 'disabled'}"
        )
        if args.use_processed_train_data:
            processed_dataset_root = Path(args.modelnet40_mild_root).expanduser().resolve()
            processed_train_dataset_full, processed_dataset_format = load_processed_modelnet40_dataset(args, labels)

    if args.use_processed_train_data:
        target_size = compute_sampled_processed_size(
            dataset_size=len(processed_train_dataset_full),
            ratio_denominator=args.mild_ratio_denominator,
        )
        processed_train_dataset = sample_processed_dataset(
            processed_train_dataset_full,
            target_size=target_size,
            seed=args.seed + 23,
        )
        train_dataset = ConcatDataset([train_dataset, processed_train_dataset])
        train_loader = build_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            workers=args.workers,
            shuffle=True,
        )

    print(f"train_samples={len(train_dataset)} | val_samples={len(test_dataset)} | num_classes={len(labels)}")
    if args.use_processed_train_data:
        processed_format_text = f" | processed_format={processed_dataset_format}" if processed_dataset_format else ""
        print(
            f"processed_train_root={processed_dataset_root}{processed_format_text} | "
            f"processed_full_samples={len(processed_train_dataset_full)} | "
            f"processed_sampled_samples={len(processed_train_dataset)} | "
            f"processed_fraction=1/{args.mild_ratio_denominator}"
        )
    print(f"device={args.device} | workers={args.workers} | batch_size={args.batch_size}")

    labels_path = output_dir / "labels.txt"
    with open(labels_path, "w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(f"{label}\n")

    if args.use_wandb and wandb is None:
        raise RuntimeError("wandb is not installed, so --use-wandb cannot be used.")

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args) | {"resolved_dataset_type": dataset_type},
        )

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_dataset, len(labels)).to(args.device)

    model = PointNet2ClsSSG(num_classes=len(labels), dropout=args.dropout).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=max(args.lr * 0.01, 1e-5)
    )

    best_acc = 0.0
    best_ckpt_path = output_dir / "pointnet2_best.pth"
    latest_ckpt_path = output_dir / "pointnet2_last.pth"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        print(f"Starting epoch {epoch}/{args.epochs}")
        train_iterator = train_loader
        if tqdm is not None:
            train_iterator = tqdm(
                train_loader,
                desc=f"train {epoch:03d}",
                leave=False,
                dynamic_ncols=True,
            )

        for points, labels_batch in train_iterator:
            points = points.to(args.device)
            labels_batch = labels_batch.to(args.device)

            optimizer.zero_grad()
            logits = model(points)
            loss = F.cross_entropy(
                logits,
                labels_batch,
                weight=class_weights,
                label_smoothing=args.label_smoothing,
            )
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels_batch.size(0)
            total_correct += (preds == labels_batch).sum().item()
            total_seen += labels_batch.size(0)

            if tqdm is not None:
                train_iterator.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{(preds == labels_batch).float().mean().item():.4f}",
                )

        train_loss = total_loss / max(total_seen, 1)
        train_acc = total_correct / max(total_seen, 1)
        print(f"Finished training epoch {epoch}, starting validation")
        val_loss, val_acc = evaluate(
            model,
            test_loader,
            args.device,
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
        )
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        print(
            f"epoch {epoch:03d} | train_loss {train_loss:.4f} | "
            f"train_acc {train_acc:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "num_points": args.num_points,
            "labels": labels,
            "dataset_type": dataset_type,
            "scanobjectnn_variant": args.scanobjectnn_variant,
            "model_type": "pointnet2_ssg",
            "use_processed_train_data": args.use_processed_train_data,
            "mild_ratio_denominator": args.mild_ratio_denominator if args.use_processed_train_data else None,
        }
        if dataset_type == "scanobjectnn":
            checkpoint["scanobjectnn_root"] = str(Path(args.scanobjectnn_root).resolve())
            if args.use_processed_train_data:
                checkpoint["scanobjectnn_mild_root"] = str(processed_dataset_root)
        else:
            checkpoint["modelnet40_root"] = str(Path(args.modelnet40_root).resolve())
            if args.use_processed_train_data:
                checkpoint["modelnet40_mild_root"] = str(processed_dataset_root)
                checkpoint["processed_dataset_format"] = processed_dataset_format
        torch.save(checkpoint, latest_ckpt_path)
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(checkpoint, best_ckpt_path)

        if args.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

    metrics_path = output_dir / "train_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    plot_paths = plot_classification_history(output_dir, history, "PointNet2 Classification")

    print(f"Training finished. Best val_acc={best_acc:.4f}")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Last checkpoint: {latest_ckpt_path}")
    print(f"Labels file: {labels_path}")
    print(f"Metrics file: {metrics_path}")
    for plot_path in plot_paths:
        print(f"plot: {plot_path}")

    if args.use_wandb:
        wandb.summary["best_checkpoint"] = str(best_ckpt_path)
        wandb.summary["labels_path"] = str(labels_path)
        wandb.summary["best_val_acc"] = best_acc
        wandb.summary["metrics_path"] = str(metrics_path)
        wandb.summary["plot_paths"] = [str(path) for path in plot_paths]
        image_logs = {
            f"plot/{Path(path).stem}": wandb.Image(str(path))
            for path in plot_paths
            if Path(path).suffix.lower() == ".png"
        }
        if image_logs:
            wandb.log(image_logs)
        wandb.finish()


if __name__ == "__main__":
    main()
