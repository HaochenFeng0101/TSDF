import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset



# python detection/train_pointnet_cls.py \
#   --dataset-type scanobjectnn \
#   --scanobjectnn-root data/ScanObjectNN \
#   --scanobjectnn-variant pb_t50_rs \
#   --epochs 150 \
#   --batch-size 32 \
#   --num-points 1024 \
#   --lr 1e-3 \
#   --weight-decay 1e-4 \
#   --label-smoothing 0.1 \
#   --feature-transform-weight 1e-3 \
#   --use-class-weights \
#   --use-wandb \
#   --wandb-project TSDF-PointNet \
#   --wandb-run-name pointnet_scanobjectnn_pb_t50_rs
  
  
REPO_ROOT = Path(__file__).resolve().parents[2]
TSDF_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSDF.detection.pointnet_model import PointNetCls, feature_transform_regularizer
from TSDF.dataset.scanobjectnn_data import (
    SCANOBJECTNN_LABELS,
    get_scanobjectnn_dataloaders,
)

try:
    import h5py
except Exception:
    h5py = None

try:
    import wandb
except Exception:
    wandb = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_points(points):
    points = points.astype(np.float32)
    centroid = points.mean(axis=0, keepdims=True)
    points = points - centroid
    scale = np.linalg.norm(points, axis=1).max()
    if scale > 0:
        points = points / scale
    return points


def sample_points(points, num_points, rng):
    if len(points) >= num_points:
        indices = rng.choice(len(points), num_points, replace=False)
    else:
        indices = rng.choice(len(points), num_points, replace=True)
    return points[indices]


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


class PointCloudClassificationDataset(Dataset):
    def __init__(
        self,
        samples,
        num_points,
        labels,
        augment=False,
        seed=0,
    ):
        self.samples = samples
        self.num_points = num_points
        self.labels = labels
        self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples)

    def _load_points(self, path):
        path = Path(path)
        if path.suffix == ".npy":
            points = np.load(path)
        elif path.suffix == ".npz":
            data = np.load(path)
            key = "points" if "points" in data else list(data.keys())[0]
            points = data[key]
        elif path.suffix in {".txt", ".pts", ".xyz"}:
            points = np.loadtxt(path)
        else:
            raise ValueError(f"Unsupported point file format: {path}")

        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Expected Nx3+ points in {path}, got shape {points.shape}")
        return points[:, :3]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        points = self._load_points(sample["path"])
        points = normalize_points(points)
        points = sample_points(points, self.num_points, self.rng)
        if self.augment:
            points = maybe_augment(points, self.rng)
        tensor = torch.from_numpy(points.T.astype(np.float32))
        label_idx = self.label_to_idx[sample["label"]]
        return tensor, label_idx


def read_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Manifest must be a JSON list of {'path': ..., 'label': ...}.")
    return data


def load_h5_samples(h5_path):
    if h5py is None:
        raise RuntimeError(
            "h5py is not installed. Install it first to train from ScanObjectNN h5 files."
        )
    samples = []
    with h5py.File(h5_path, "r") as handle:
        data = np.asarray(handle["data"])
        labels = np.asarray(handle["label"]).reshape(-1)
        for idx in range(len(labels)):
            samples.append({"points": data[idx][:, :3], "label_idx": int(labels[idx])})
    return samples


class H5ClassificationDataset(Dataset):
    def __init__(self, samples, num_points, labels, augment=False, seed=0):
        self.samples = samples
        self.num_points = num_points
        self.labels = labels
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        points = normalize_points(sample["points"])
        points = sample_points(points, self.num_points, self.rng)
        if self.augment:
            points = maybe_augment(points, self.rng)
        tensor = torch.from_numpy(points.T.astype(np.float32))
        return tensor, sample["label_idx"]


def build_dir_splits(data_root):
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    labels = sorted(path.name for path in data_root.iterdir() if path.is_dir())
    if not labels:
        raise ValueError(f"No class folders found in {data_root}")

    train_samples = []
    test_samples = []
    exts = {".npy", ".npz", ".txt", ".pts", ".xyz"}
    for label in labels:
        train_dir = data_root / label / "train"
        test_dir = data_root / label / "test"
        if not train_dir.exists() or not test_dir.exists():
            raise ValueError(
                f"Expected {train_dir} and {test_dir} for class '{label}'"
            )

        for path in sorted(train_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in exts:
                train_samples.append({"path": str(path), "label": label})
        for path in sorted(test_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in exts:
                test_samples.append({"path": str(path), "label": label})

    return labels, train_samples, test_samples


def evaluate(
    model,
    dataloader,
    device,
    class_weights=None,
    label_smoothing=0.0,
    feature_transform_weight=1e-3,
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    with torch.no_grad():
        for points, labels in dataloader:
            points = points.to(device)
            labels = labels.to(device)
            logits, trans_feat = model(points)
            loss = F.cross_entropy(
                logits,
                labels,
                weight=class_weights,
                label_smoothing=label_smoothing,
            )
            loss = loss + feature_transform_weight * feature_transform_regularizer(trans_feat)
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_seen += labels.size(0)
    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


def compute_class_weights(dataset, num_classes):
    if hasattr(dataset, "labels"):
        raw_labels = np.asarray(dataset.labels, dtype=np.int64)
    else:
        raw_labels = np.asarray([sample["label_idx"] for sample in dataset.samples], dtype=np.int64)
    counts = np.bincount(raw_labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


def main():
    parser = argparse.ArgumentParser(
        description="Train a PointNet classifier checkpoint for object point clouds."
    )
    parser.add_argument(
        "--dataset-type",
        choices=["dir", "h5", "scanobjectnn"],
        default="dir",
        help="Use a folder dataset or ScanObjectNN-style h5 files.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Directory dataset root. Expected layout: data_root/class_name/train/*.npy and test/*.npy",
    )
    parser.add_argument("--train-h5", default=None, help="Training h5 file for h5 mode.")
    parser.add_argument("--test-h5", default=None, help="Test h5 file for h5 mode.")
    parser.add_argument(
        "--labels",
        default=None,
        help="Optional label file for h5 mode, one class per line. For dir mode it is written automatically.",
    )
    parser.add_argument(
        "--scanobjectnn-root",
        default=str(TSDF_ROOT / "data" / "ScanObjectNN"),
        help="Root directory created by download_scanobjectnn.py",
    )
    parser.add_argument(
        "--scanobjectnn-variant",
        default="pb_t50_rs",
        help="Variant for ScanObjectNN: pb_t50_rs, pb_t50_r, pb_t25, pb_t25_r, obj_bg, obj_only",
    )
    parser.add_argument(
        "--scanobjectnn-no-bg",
        action="store_true",
        help="Use main_split_nobg instead of main_split for ScanObjectNN.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--feature-transform-weight", type=float, default=1e-3)
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use inverse-frequency class weights in the classification loss.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=str(TSDF_ROOT / "model" / "pointnet"),
        help="Where to save checkpoints.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Log training metrics to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project",
        default="TSDF-PointNet",
        help="wandb project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional wandb run name.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_type == "dir":
        if args.data_root is None:
            raise ValueError("--data-root is required for --dataset-type dir")
        labels, train_samples, test_samples = build_dir_splits(args.data_root)
        train_dataset = PointCloudClassificationDataset(
            train_samples, args.num_points, labels, augment=True, seed=args.seed
        )
        test_dataset = PointCloudClassificationDataset(
            test_samples, args.num_points, labels, augment=False, seed=args.seed + 1
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            drop_last=False,
        )
    elif args.dataset_type == "h5":
        if args.train_h5 is None or args.test_h5 is None or args.labels is None:
            raise ValueError(
                "--train-h5, --test-h5, and --labels are required for --dataset-type h5"
            )
        with open(args.labels, "r", encoding="utf-8") as handle:
            labels = [line.strip() for line in handle if line.strip()]
        train_samples = load_h5_samples(args.train_h5)
        test_samples = load_h5_samples(args.test_h5)
        train_dataset = H5ClassificationDataset(
            train_samples, args.num_points, labels, augment=True, seed=args.seed
        )
        test_dataset = H5ClassificationDataset(
            test_samples, args.num_points, labels, augment=False, seed=args.seed + 1
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            drop_last=False,
        )
    else:
        if args.scanobjectnn_root is None:
            raise ValueError(
                "--scanobjectnn-root is required for --dataset-type scanobjectnn"
            )
        labels = SCANOBJECTNN_LABELS
        train_dataset, test_dataset, train_loader, test_loader = (
            get_scanobjectnn_dataloaders(
                root=args.scanobjectnn_root,
                variant=args.scanobjectnn_variant,
                batch_size=args.batch_size,
                num_points=args.num_points,
                workers=args.workers,
                use_background=not args.scanobjectnn_no_bg,
                seed=args.seed,
            )
        )

    labels_path = output_dir / "labels.txt"
    with open(labels_path, "w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(f"{label}\n")

    if args.use_wandb and wandb is None:
        raise RuntimeError(
            "wandb is not installed in this environment. Install it or omit --use-wandb."
        )

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
        wandb.config.update(
            {
                "num_classes": len(labels),
                "labels_path": str(labels_path),
            },
            allow_val_change=True,
        )

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_dataset, len(labels)).to(args.device)

    model = PointNetCls(k=len(labels), dropout=args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=max(args.lr * 0.01, 1e-5)
    )

    best_acc = 0.0
    best_ckpt_path = output_dir / "pointnet_best.pth"
    latest_ckpt_path = output_dir / "pointnet_last.pth"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for points, labels_batch in train_loader:
            points = points.to(args.device)
            labels_batch = labels_batch.to(args.device)

            optimizer.zero_grad()
            logits, trans_feat = model(points)
            loss = F.cross_entropy(
                logits,
                labels_batch,
                weight=class_weights,
                label_smoothing=args.label_smoothing,
            )
            loss = loss + (
                args.feature_transform_weight * feature_transform_regularizer(trans_feat)
            )
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels_batch.size(0)
            total_correct += (preds == labels_batch).sum().item()
            total_seen += labels_batch.size(0)

        train_loss = total_loss / max(total_seen, 1)
        train_acc = total_correct / max(total_seen, 1)
        val_loss, val_acc = evaluate(
            model,
            test_loader,
            args.device,
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
            feature_transform_weight=args.feature_transform_weight,
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

        if args.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                    "best_val_acc": max(best_acc, val_acc),
                }
            )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "labels": labels,
            "num_points": args.num_points,
            "dropout": args.dropout,
            "label_smoothing": args.label_smoothing,
            "feature_transform_weight": args.feature_transform_weight,
            "val_acc": val_acc,
        }
        torch.save(ckpt, latest_ckpt_path)
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(ckpt, best_ckpt_path)

    metrics_path = output_dir / "train_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    if args.use_wandb:
        wandb.summary["best_checkpoint"] = str(best_ckpt_path)
        wandb.summary["labels_path"] = str(labels_path)
        wandb.summary["metrics_path"] = str(metrics_path)
        wandb.summary["best_val_acc"] = best_acc
        wandb.finish()

    print(f"best checkpoint: {best_ckpt_path}")
    print(f"labels file: {labels_path}")
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
