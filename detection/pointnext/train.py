import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[3]
TSDF_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSDF.dataset.extra_object_data import (
    ModelNet40H5Dataset,
    ModelNet40OffDataset,
    extra_object_root_exists,
    get_modelnet40_with_extra_dataloaders,
    get_scanobjectnn_with_extra_dataloaders,
    load_point_file,
    modelnet40_root_exists,
)
from TSDF.dataset.scanobjectnn_data import SCANOBJECTNN_LABELS, ScanObjectNNDataset
from TSDF.detection.pointnext.pointnext_cls import PointNeXtSmallCls

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


def scanobjectnn_is_ready(root):
    root = Path(root)
    return root.exists() and ((root / "h5_files" / "main_split").exists() or (root / "main_split").exists())


def resolve_dataset_type(args):
    if args.dataset_type != "auto":
        return args.dataset_type
    if scanobjectnn_is_ready(args.scanobjectnn_root):
        return "scanobjectnn"
    if modelnet40_root_exists(args.modelnet40_root):
        return "modelnet40"
    raise RuntimeError(
        "No usable dataset found. Provide --dataset-type explicitly or prepare ScanObjectNN/ModelNet40 data."
    )


def build_dataloader(dataset, batch_size, workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=False,
    )


class ExtraPointCloudDataset(Dataset):
    def __init__(self, point_cloud_path, label_idx, num_points, repeat=1, augment=True, seed=0):
        self.point_cloud_path = str(Path(point_cloud_path).expanduser().resolve())
        self.label_idx = int(label_idx)
        self.num_points = int(num_points)
        self.repeat = max(int(repeat), 1)
        self.augment = augment
        self.seed = int(seed)

        from TSDF.dataset.scanobjectnn_data import maybe_augment, normalize_points

        self._maybe_augment = maybe_augment
        self._normalize_points = normalize_points
        self.points = load_point_file(self.point_cloud_path)

    def __len__(self):
        return self.repeat

    def __getitem__(self, idx):
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
        return torch.from_numpy(points.T.astype(np.float32)), self.label_idx


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

    if isinstance(dataset, ExtraPointCloudDataset):
        return [dataset.label_idx] * len(dataset)

    raise ValueError(f"Cannot infer label indices from dataset type: {type(dataset).__name__}")


def compute_class_weights_for_dataset(dataset, num_classes, device):
    raw_labels = np.asarray(collect_label_indices(dataset), dtype=np.int64)
    counts = np.bincount(raw_labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return torch.from_numpy(weights).to(device)


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
        dataset = ModelNet40H5Dataset(
            root=args.modelnet40_mild_root,
            split="train",
            num_points=args.num_points,
            augment=True,
            seed=args.seed + 11,
        )
        processed_format = "h5"
    else:
        dataset = ModelNet40OffDataset(
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


def evaluate(
    model,
    dataloader,
    device,
    class_weights=None,
    label_smoothing=0.0,
):
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


def main():
    parser = argparse.ArgumentParser(
        description="Train a PointNeXt-Small classifier on ScanObjectNN or ModelNet40 with optional extra/mild data."
    )
    parser.add_argument(
        "--dataset-type",
        choices=["auto", "scanobjectnn", "modelnet40"],
        default="auto",
        help="auto prefers ScanObjectNN and falls back to ModelNet40.",
    )
    parser.add_argument("--scanobjectnn-root", default=str(TSDF_ROOT / "data" / "ScanObjectNN"))
    parser.add_argument("--scanobjectnn-variant", default="pb_t50_rs")
    parser.add_argument("--scanobjectnn-no-bg", action="store_true")
    parser.add_argument(
        "--extra-object-root",
        default=str(TSDF_ROOT / "data" / "extra_object"),
        help="Optional extra object directory to merge into train/test.",
    )
    parser.add_argument("--no-extra-object-data", action="store_true")
    parser.add_argument("--modelnet40-root", default=str(TSDF_ROOT / "data" / "ModelNet40"))
    parser.add_argument("--use-processed-train-data", action="store_true")
    parser.add_argument("--scanobjectnn-mild-root", default=str(TSDF_ROOT / "data" / "ScanObjectNN_mild"))
    parser.add_argument("--modelnet40-mild-root", default=str(TSDF_ROOT / "data" / "ModelNet40_mild"))
    parser.add_argument("--mild-ratio-denominator", type=int, default=5)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--model-type", choices=["pointnext_small"], default="pointnext_small")
    parser.add_argument("--input-channels", type=int, default=3)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--extra-train-sample", default=None)
    parser.add_argument("--extra-train-label", default=None)
    parser.add_argument("--extra-train-repeat", type=int, default=64)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-dir", default=str(TSDF_ROOT / "model" / "pointnext"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="TSDF-PointNeXt")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_type = resolve_dataset_type(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_train_dataset_full = None
    processed_train_dataset = None
    processed_dataset_root = None
    processed_dataset_format = None
    extra_train_dataset = None

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
            f"dataset=ScanObjectNN | variant={args.scanobjectnn_variant} | "
            f"use_background={not args.scanobjectnn_no_bg} | "
            f"extra_object_root={args.extra_object_root if not args.no_extra_object_data else 'disabled'}"
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
            f"dataset=ModelNet40 | "
            f"extra_object_root={args.extra_object_root if not args.no_extra_object_data else 'disabled'} | "
            f"extra_object_exists={extra_object_root_exists(args.extra_object_root)}"
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

    if args.extra_train_sample:
        if not args.extra_train_label:
            raise ValueError("--extra-train-label is required when --extra-train-sample is set.")
        if args.extra_train_label not in labels:
            raise ValueError(f"Unknown extra train label '{args.extra_train_label}'. Available labels: {labels}")
        extra_train_dataset = ExtraPointCloudDataset(
            point_cloud_path=args.extra_train_sample,
            label_idx=labels.index(args.extra_train_label),
            num_points=args.num_points,
            repeat=args.extra_train_repeat,
            augment=True,
            seed=args.seed + 1000,
        )
        train_dataset = ConcatDataset([train_dataset, extra_train_dataset])
        train_loader = build_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            workers=args.workers,
            shuffle=True,
        )

    print(f"train_samples={len(train_dataset)} | test_samples={len(test_dataset)} | num_classes={len(labels)}")
    if args.use_processed_train_data:
        processed_format_text = f" | processed_format={processed_dataset_format}" if processed_dataset_format else ""
        print(
            f"processed_train_root={processed_dataset_root}{processed_format_text} | "
            f"processed_full_samples={len(processed_train_dataset_full)} | "
            f"processed_sampled_samples={len(processed_train_dataset)} | "
            f"processed_fraction=1/{args.mild_ratio_denominator}"
        )
    if args.extra_train_sample:
        print(
            f"extra_train_sample={Path(args.extra_train_sample).expanduser().resolve()} | "
            f"extra_train_label={args.extra_train_label} | "
            f"extra_train_repeat={args.extra_train_repeat} | "
            f"extra_train_samples={len(extra_train_dataset)}"
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

    model = PointNeXtSmallCls(
        num_classes=len(labels),
        input_channels=args.input_channels,
        dropout=args.dropout,
    ).to(args.device)
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
        optimizer,
        T_max=args.epochs,
        eta_min=max(args.lr * 0.01, 1e-5),
    )

    use_amp = args.amp and str(args.device).startswith("cuda")
    amp_device_type = "cuda" if str(args.device).startswith("cuda") else "cpu"
    scaler = torch.amp.GradScaler(amp_device_type, enabled=use_amp)

    best_acc = 0.0
    best_ckpt_path = output_dir / "pointnext_best.pth"
    latest_ckpt_path = output_dir / "pointnext_last.pth"
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
                    logits,
                    labels_batch,
                    weight=class_weights,
                    label_smoothing=args.label_smoothing,
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

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "labels": labels,
            "num_points": args.num_points,
            "dropout": args.dropout,
            "label_smoothing": args.label_smoothing,
            "val_acc": val_acc,
            "task": "classification",
            "dataset_type": dataset_type,
            "scanobjectnn_variant": args.scanobjectnn_variant,
            "use_background": not args.scanobjectnn_no_bg,
            "model_type": args.model_type,
            "use_processed_train_data": args.use_processed_train_data,
            "mild_ratio_denominator": args.mild_ratio_denominator if args.use_processed_train_data else None,
            "extra_train_sample": str(Path(args.extra_train_sample).expanduser().resolve()) if args.extra_train_sample else None,
            "extra_train_label": args.extra_train_label,
            "extra_train_repeat": args.extra_train_repeat if args.extra_train_sample else None,
        }
        if dataset_type == "scanobjectnn":
            ckpt["scanobjectnn_root"] = str(Path(args.scanobjectnn_root).resolve())
            if args.use_processed_train_data:
                ckpt["scanobjectnn_mild_root"] = str(processed_dataset_root)
        else:
            ckpt["modelnet40_root"] = str(Path(args.modelnet40_root).resolve())
            if args.use_processed_train_data:
                ckpt["modelnet40_mild_root"] = str(processed_dataset_root)
                ckpt["processed_dataset_format"] = processed_dataset_format

        torch.save(ckpt, latest_ckpt_path)
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(ckpt, best_ckpt_path)

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
    print(f"last checkpoint: {latest_ckpt_path}")
    print(f"labels file: {labels_path}")
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
