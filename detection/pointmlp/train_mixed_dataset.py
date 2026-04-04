import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

"""
python detection/pointmlp/train_mixed_dataset.py

python detection/pointmlp/train_mixed_dataset.py `
  --dataset-type scanobjectnn `
  --scanobjectnn-root data/ScanObjectNN `
  --scanobjectnn-mild-root data/ScanObjectNN_mild `
  --scanobjectnn-variant pb_t50_rs `
  --batch-size 16 `
  --num-points 2048 `
  --amp `
  --use-class-weights
  --overwrite

python detection/pointmlp/train_mixed_dataset.py `
  --dataset-type modelnet40 `
  --modelnet40-root data/ModelNet40 `
  --modelnet40-mild-root data/ModelNet40_mild `
  --batch-size 16 `
  --num-points 2048 `
  --amp `
  --use-class-weights `
  --overwrite
"""


REPO_ROOT = Path(__file__).resolve().parents[3]
TSDF_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSDF.dataset.modelnet40_data import ModelNet40Dataset
from TSDF.dataset.scanobjectnn_data import SCANOBJECTNN_LABELS, ScanObjectNNDataset
from TSDF.detection.pointmlp.pointmlp_cls import PointMLPCls
from TSDF.detection.train_pointnet_cls import (
    H5ClassificationDataset,
    load_h5_samples,
    load_point_cloud_file,
    set_seed,
)

try:
    import wandb
except Exception:
    wandb = None


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


class ExtraPointCloudDataset(Dataset):
    def __init__(self, point_cloud_path, label_idx, num_points, repeat=1, augment=True, seed=0):
        self.point_cloud_path = str(Path(point_cloud_path).expanduser().resolve())
        self.label_idx = int(label_idx)
        self.num_points = num_points
        self.repeat = max(int(repeat), 1)
        self.augment = augment
        self.seed = seed

        from TSDF.dataset.scanobjectnn_data import maybe_augment, normalize_points

        self._maybe_augment = maybe_augment
        self._normalize_points = normalize_points
        self.points = load_point_cloud_file(self.point_cloud_path)

    def __len__(self):
        return self.repeat

    def __getitem__(self, idx):
        import numpy as np

        rng = np.random.default_rng(self.seed + idx)
        points = self.points.copy()
        points = self._normalize_points(points)
        if len(points) >= self.num_points:
            choice = rng.choice(len(points), self.num_points, replace=False)
        else:
            choice = rng.choice(len(points), self.num_points, replace=True)
        points = points[choice]
        if self.augment:
            points = self._maybe_augment(points, rng)
        return torch.from_numpy(points.T.astype("float32")), self.label_idx


def collect_label_indices(dataset):
    if isinstance(dataset, ConcatDataset):
        indices = []
        for subdataset in dataset.datasets:
            indices.extend(collect_label_indices(subdataset))
        return indices

    if hasattr(dataset, "label_to_idx") and hasattr(dataset, "samples"):
        return [dataset.label_to_idx[sample["label"]] for sample in dataset.samples]

    if hasattr(dataset, "samples") and len(dataset.samples) > 0 and "label_idx" in dataset.samples[0]:
        return [int(sample["label_idx"]) for sample in dataset.samples]

    if hasattr(dataset, "labels"):
        labels = getattr(dataset, "labels")
        if len(labels) > 0:
            first = labels[0]
            if isinstance(first, int) or hasattr(first, "item"):
                return [int(x) for x in labels]

    if isinstance(dataset, ExtraPointCloudDataset):
        return [dataset.label_idx] * len(dataset)

    raise ValueError(f"Cannot infer label indices from dataset type: {type(dataset).__name__}")


def compute_class_weights_for_dataset(dataset, num_classes, device):
    counts = torch.bincount(torch.tensor(collect_label_indices(dataset), dtype=torch.long), minlength=num_classes).float()
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return weights.to(device)


def get_mixed_scanobjectnn_dataloaders(
    root,
    mild_root,
    variant="pb_t50_rs",
    batch_size=32,
    num_points=1024,
    workers=4,
    use_background=True,
    seed=0,
):
    train_dataset_raw = ScanObjectNNDataset(
        root=root,
        split="train",
        variant=variant,
        num_points=num_points,
        use_background=use_background,
        normalize=True,
        augment=True,
        seed=seed,
    )
    train_dataset_mild = ScanObjectNNDataset(
        root=mild_root,
        split="train",
        variant=variant,
        num_points=num_points,
        use_background=use_background,
        normalize=True,
        augment=True,
        seed=seed + 11,
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

    train_dataset = ConcatDataset([train_dataset_raw, train_dataset_mild])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        drop_last=False,
    )
    return train_dataset_raw, train_dataset_mild, train_dataset, test_dataset, train_loader, test_loader


def resolve_modelnet40_h5_root(root):
    root = Path(root)
    candidates = [root, root / "modelnet40_ply_hdf5_2048"]
    for candidate in candidates:
        if (candidate / "train_files.txt").exists() and (candidate / "shape_names.txt").exists():
            return candidate

    if root.exists():
        for subdir in sorted(path for path in root.rglob("*") if path.is_dir()):
            if (subdir / "train_files.txt").exists() and (subdir / "shape_names.txt").exists():
                return subdir
    raise FileNotFoundError(f"Could not find ModelNet40 h5 root under {root}")


def modelnet40_root_looks_like_h5(root):
    try:
        resolve_modelnet40_h5_root(root)
        return True
    except Exception:
        return False


def read_modelnet40_labels(root):
    root = Path(root)
    for candidate in (
        root / "shape_names.txt",
        root / "modelnet40_shape_names.txt",
        root / "labels.txt",
    ):
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as handle:
                labels = [line.strip() for line in handle if line.strip()]
            if labels:
                return labels
    raise FileNotFoundError(f"Could not find ModelNet40 label names under {root}")


def load_modelnet40_h5_dataset(root, split, num_points, augment, seed):
    h5_root = resolve_modelnet40_h5_root(root)
    labels = read_modelnet40_labels(h5_root)
    split_filename = "train_files.txt" if split == "train" else "test_files.txt"
    split_path = h5_root / split_filename
    if not split_path.exists():
        raise FileNotFoundError(f"Could not find {split_path}")

    samples = []
    with open(split_path, "r", encoding="utf-8") as handle:
        relative_paths = [line.strip() for line in handle if line.strip()]

    for relpath in relative_paths:
        relpath = relpath.replace("\\", "/")
        relpath_obj = Path(relpath)
        candidates = [h5_root / relpath_obj.name, h5_root / relpath_obj]
        h5_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if h5_path is None:
            raise FileNotFoundError(f"Could not find H5 file listed in {split_path}: {relpath}")
        samples.extend(load_h5_samples(h5_path))

    dataset = H5ClassificationDataset(samples, num_points, labels, augment=augment, seed=seed)
    return labels, dataset


def get_mixed_modelnet40_dataloaders(
    root,
    mild_root,
    batch_size=32,
    num_points=1024,
    workers=4,
    seed=0,
    sample_method="surface",
):
    if modelnet40_root_looks_like_h5(root):
        labels, train_dataset_raw = load_modelnet40_h5_dataset(
            root=root,
            split="train",
            num_points=num_points,
            augment=True,
            seed=seed,
        )
        test_labels, test_dataset = load_modelnet40_h5_dataset(
            root=root,
            split="test",
            num_points=num_points,
            augment=False,
            seed=seed + 1,
        )
    else:
        train_dataset_raw = ModelNet40Dataset(
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
        labels = train_dataset_raw.labels
        test_labels = test_dataset.labels

    mild_labels, train_dataset_mild = load_modelnet40_h5_dataset(
        root=mild_root,
        split="train",
        num_points=num_points,
        augment=True,
        seed=seed + 11,
    )

    if labels != test_labels:
        raise ValueError("Raw ModelNet40 train/test labels do not match.")
    if labels != mild_labels:
        raise ValueError("Raw ModelNet40 labels do not match processed ModelNet40 labels.")

    train_dataset = ConcatDataset([train_dataset_raw, train_dataset_mild])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        drop_last=False,
    )
    return labels, train_dataset_raw, train_dataset_mild, train_dataset, test_dataset, train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(
        description="Train a PointMLP classifier on original + mildly processed ScanObjectNN or ModelNet40."
    )
    parser.add_argument(
        "--dataset-type",
        choices=["scanobjectnn", "modelnet40"],
        default="scanobjectnn",
        help="Which dataset family to use for mixed training.",
    )
    parser.add_argument(
        "--scanobjectnn-root",
        default=str(TSDF_ROOT / "data" / "ScanObjectNN"),
        help="Root directory for the original ScanObjectNN dataset.",
    )
    parser.add_argument(
        "--scanobjectnn-mild-root",
        default=str(TSDF_ROOT / "data" / "ScanObjectNN_mild"),
        help="Root directory for the processed ScanObjectNN dataset.",
    )
    parser.add_argument(
        "--scanobjectnn-variant",
        default="pb_t50_rs",
        help="Variant for ScanObjectNN: pb_t50_rs, pb_t50_r, pb_t25, pb_t25_r, obj_bg, obj_only",
    )
    parser.add_argument("--scanobjectnn-no-bg", action="store_true")
    parser.add_argument(
        "--modelnet40-root",
        default=str(TSDF_ROOT / "data" / "ModelNet40"),
        help="Root directory for the original ModelNet40 dataset.",
    )
    parser.add_argument(
        "--modelnet40-mild-root",
        default=str(TSDF_ROOT / "data" / "ModelNet40_mild"),
        help="Root directory for the processed ModelNet40 dataset.",
    )
    parser.add_argument(
        "--modelnet40-sample-method",
        choices=["surface", "vertex"],
        default="surface",
        help="How to sample points when the original ModelNet40 root is OFF-based.",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--model-type", choices=["pointmlp", "pointmlpelite"], default="pointmlp")
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument(
        "--extra-train-sample",
        default=str(TSDF_ROOT / "3d_construction" / "outputs" / "fr3_office_main_chair_yolo.pcd"),
        help="Optional point cloud file to inject into the training set only.",
    )
    parser.add_argument(
        "--extra-train-label",
        default="chair",
        help="Label name for --extra-train-sample.",
    )
    parser.add_argument(
        "--extra-train-repeat",
        type=int,
        default=64,
        help="How many times to repeat the injected point cloud in the training set.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision to reduce GPU memory usage.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(TSDF_ROOT / "model" / "pointmlp_mixed"),
        help="Where to save PointMLP checkpoints.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="TSDF-PointMLP")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_type == "scanobjectnn":
        labels = SCANOBJECTNN_LABELS
        (
            train_dataset_raw,
            train_dataset_mild,
            train_dataset,
            test_dataset,
            train_loader,
            test_loader,
        ) = get_mixed_scanobjectnn_dataloaders(
            root=args.scanobjectnn_root,
            mild_root=args.scanobjectnn_mild_root,
            variant=args.scanobjectnn_variant,
            batch_size=args.batch_size,
            num_points=args.num_points,
            workers=args.workers,
            use_background=not args.scanobjectnn_no_bg,
            seed=args.seed,
        )
        dataset_summary = (
            f"dataset=ScanObjectNN(raw+mild) | variant={args.scanobjectnn_variant} | "
            f"use_background={not args.scanobjectnn_no_bg}"
        )
    else:
        (
            labels,
            train_dataset_raw,
            train_dataset_mild,
            train_dataset,
            test_dataset,
            train_loader,
            test_loader,
        ) = get_mixed_modelnet40_dataloaders(
            root=args.modelnet40_root,
            mild_root=args.modelnet40_mild_root,
            batch_size=args.batch_size,
            num_points=args.num_points,
            workers=args.workers,
            seed=args.seed,
            sample_method=args.modelnet40_sample_method,
        )
        raw_format = "h5" if modelnet40_root_looks_like_h5(args.modelnet40_root) else "off"
        dataset_summary = (
            f"dataset=ModelNet40(raw+mild) | raw_format={raw_format} | "
            f"sample_method={args.modelnet40_sample_method if raw_format == 'off' else 'h5'}"
        )

    if args.extra_train_sample:
        if args.extra_train_label not in labels:
            raise ValueError(
                f"Unknown extra train label '{args.extra_train_label}'. Available labels: {labels}"
            )
        extra_dataset = ExtraPointCloudDataset(
            point_cloud_path=args.extra_train_sample,
            label_idx=labels.index(args.extra_train_label),
            num_points=args.num_points,
            repeat=args.extra_train_repeat,
            augment=True,
            seed=args.seed + 1000,
        )
        train_dataset = ConcatDataset([train_dataset, extra_dataset])
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=False,
        )
        print(
            "extra training sample injected: "
            f"path={Path(args.extra_train_sample).resolve()} "
            f"label={args.extra_train_label} "
            f"repeat={args.extra_train_repeat}"
        )

    print(dataset_summary)
    print(f"raw_train_samples={len(train_dataset_raw)} | mild_train_samples={len(train_dataset_mild)}")
    print(
        f"mixed_train_samples={len(train_dataset)} | test_samples={len(test_dataset)} | "
        f"num_classes={len(labels)}"
    )

    labels_path = output_dir / "labels.txt"
    with open(labels_path, "w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(f"{label}\n")

    if args.use_wandb and wandb is None:
        raise RuntimeError("wandb is not installed in this environment. Install it or omit --use-wandb.")
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights_for_dataset(train_dataset, len(labels), args.device)

    model = PointMLPCls(k=len(labels), num_points=args.num_points, model_type=args.model_type).to(args.device)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=max(args.lr * 0.01, 1e-5)
    )
    use_amp = args.amp and str(args.device).startswith("cuda")
    amp_device_type = "cuda" if str(args.device).startswith("cuda") else "cpu"
    scaler = torch.amp.GradScaler(amp_device_type, enabled=use_amp)

    best_acc = 0.0
    best_ckpt_path = output_dir / "pointmlp_best_weights.pth"
    latest_ckpt_path = output_dir / "pointmlp_last_weights.pth"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        iterator = train_loader
        if tqdm is not None:
            iterator = tqdm(train_loader, desc=f"train {epoch:03d}", leave=False, dynamic_ncols=True)

        for points, labels_batch in iterator:
            points = points.to(args.device)
            labels_batch = labels_batch.to(args.device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
                logits = model(points)
                loss = F.cross_entropy(
                    logits, labels_batch, weight=class_weights, label_smoothing=args.label_smoothing
                )
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels_batch.size(0)
            total_correct += (preds == labels_batch).sum().item()
            total_seen += labels_batch.size(0)
            if tqdm is not None:
                iterator.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(total_seen, 1)
        train_acc = total_correct / max(total_seen, 1)
        val_loss, val_acc = evaluate(
            model, test_loader, args.device, class_weights=class_weights, label_smoothing=args.label_smoothing
        )
        scheduler.step()

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}
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
            "labels": labels,
            "num_points": args.num_points,
            "model_type": args.model_type,
            "val_acc": val_acc,
            "task": "classification",
            "dataset": f"{args.dataset_type}(raw+mild)",
            "dataset_type": args.dataset_type,
        }
        if args.dataset_type == "scanobjectnn":
            ckpt["scanobjectnn_variant"] = args.scanobjectnn_variant
            ckpt["scanobjectnn_root"] = str(Path(args.scanobjectnn_root).resolve())
            ckpt["scanobjectnn_mild_root"] = str(Path(args.scanobjectnn_mild_root).resolve())
            ckpt["use_background"] = not args.scanobjectnn_no_bg
        else:
            ckpt["modelnet40_root"] = str(Path(args.modelnet40_root).resolve())
            ckpt["modelnet40_mild_root"] = str(Path(args.modelnet40_mild_root).resolve())
            ckpt["modelnet40_sample_method"] = args.modelnet40_sample_method
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
