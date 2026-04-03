from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[3]
TSDF_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSDF.dataset.scanobjectnn_data import SCANOBJECTNN_LABELS, get_scanobjectnn_dataloaders
from TSDF.detection.pointnet2.train import (
    ModelNet40H5Dataset,
    ModelNet40OffDataset,
    compute_class_weights,
    modelnet40_is_ready,
    scanobjectnn_is_ready,
    set_seed,
)
from TSDF.detection.pointnext.pointnext_cls import PointNeXtSmallCls
from TSDF.detection.training_plots import plot_classification_history

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    import wandb
except Exception:
    wandb = None


def resolve_dataset_type(args):
    if args.dataset_type != "auto":
        return args.dataset_type
    if scanobjectnn_is_ready(args.scanobjectnn_root):
        return "scanobjectnn"
    if modelnet40_is_ready(args.modelnet40_root):
        return "modelnet40"
    raise RuntimeError(
        "No usable dataset found. Provide --dataset-type explicitly or prepare ScanObjectNN/ModelNet40 data."
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


def build_output_dir(output_dir: str, timestamp_output: bool) -> Path:
    root = Path(output_dir)
    if timestamp_output:
        root = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_modelnet_dataloaders(args):
    modelnet_root = Path(args.modelnet40_root)
    h5_ready = (
        (modelnet_root / "modelnet40_ply_hdf5_2048" / "train_files.txt").exists()
        or (modelnet_root / "train_files.txt").exists()
    )

    if h5_ready:
        train_dataset = ModelNet40H5Dataset(
            root=args.modelnet40_root,
            split="train",
            num_points=args.num_points,
            augment=True,
            seed=args.seed,
        )
        test_dataset = ModelNet40H5Dataset(
            root=args.modelnet40_root,
            split="test",
            num_points=args.num_points,
            augment=False,
            seed=args.seed + 1,
        )
    else:
        train_dataset = ModelNet40OffDataset(
            root=args.modelnet40_root,
            split="train",
            num_points=args.num_points,
            augment=True,
            seed=args.seed,
        )
        test_dataset = ModelNet40OffDataset(
            root=args.modelnet40_root,
            split="test",
            num_points=args.num_points,
            augment=False,
            seed=args.seed + 1,
        )

    labels = train_dataset.labels
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
    return labels, train_dataset, test_dataset, train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(
        description="Train PointNeXt-Small for point cloud classification (ScanObjectNN or ModelNet40)."
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
        help="Use ScanObjectNN no-background split.",
    )
    parser.add_argument(
        "--modelnet40-root",
        default=str(TSDF_ROOT / "data" / "ModelNet40"),
        help="ModelNet40 root directory.",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Enable AMP on CUDA.")
    parser.add_argument("--model-type", choices=["pointnext_small"], default="pointnext_small")
    parser.add_argument(
        "--output-dir",
        default=str(TSDF_ROOT / "model" / "pointnext"),
        help="Checkpoint output directory.",
    )
    parser.add_argument(
        "--timestamp-output",
        action="store_true",
        help="Append YYYYMMDD_HHMMSS under output-dir for this run.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="TSDF-PointNeXt")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_type = resolve_dataset_type(args)
    output_dir = build_output_dir(args.output_dir, args.timestamp_output)

    run_config = vars(args).copy()
    run_config["resolved_dataset_type"] = dataset_type
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    if dataset_type == "scanobjectnn":
        labels = SCANOBJECTNN_LABELS
        train_dataset, test_dataset, train_loader, test_loader = get_scanobjectnn_dataloaders(
            root=args.scanobjectnn_root,
            variant=args.scanobjectnn_variant,
            batch_size=args.batch_size,
            num_points=args.num_points,
            workers=args.workers,
            use_background=not args.scanobjectnn_no_bg,
            seed=args.seed,
        )
    else:
        labels, train_dataset, test_dataset, train_loader, test_loader = build_modelnet_dataloaders(args)

    print(f"Using dataset_type={dataset_type}")
    print(f"train_samples={len(train_dataset)} | val_samples={len(test_dataset)} | num_classes={len(labels)}")
    print(f"device={args.device} | workers={args.workers} | batch_size={args.batch_size}")

    labels_path = output_dir / "labels.txt"
    with open(labels_path, "w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(f"{label}\n")

    if args.use_wandb and wandb is None:
        raise RuntimeError("wandb is not installed, so --use-wandb cannot be used.")
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=run_config)

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_dataset, len(labels)).to(args.device)

    model = PointNeXtSmallCls(num_classes=len(labels), input_channels=3, dropout=args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=max(args.lr * 0.01, 1e-5)
    )

    use_amp = args.amp and str(args.device).startswith("cuda")
    amp_device_type = "cuda" if str(args.device).startswith("cuda") else "cpu"
    scaler = torch.amp.GradScaler(amp_device_type, enabled=use_amp)

    best_acc = 0.0
    best_ckpt_path = output_dir / "pointnext_best.pth"
    last_ckpt_path = output_dir / "pointnext_last.pth"
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

        checkpoint = {
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
        }

        torch.save(checkpoint, last_ckpt_path)
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
                    "best_val_acc": max(best_acc, val_acc),
                }
            )

    metrics_path = output_dir / "train_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    plot_paths = plot_classification_history(output_dir, history, "PointNeXt Classification")

    print(f"Training finished. Best val_acc={best_acc:.4f}")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Last checkpoint: {last_ckpt_path}")
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
