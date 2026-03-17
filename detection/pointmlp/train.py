import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


REPO_ROOT = Path(__file__).resolve().parents[3]
TSDF_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSDF.dataset.scanobjectnn_data import SCANOBJECTNN_LABELS, get_scanobjectnn_dataloaders
from TSDF.detection.pointmlp.model import PointMLPCls
from TSDF.detection.train_pointnet_cls import (
    H5ClassificationDataset,
    PointCloudClassificationDataset,
    build_dir_splits,
    compute_class_weights,
    load_h5_samples,
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
    with torch.no_grad():
        for points, labels in dataloader:
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
        description="Train a PointMLP classifier for object point clouds."
    )
    parser.add_argument("--dataset-type", choices=["dir", "h5", "scanobjectnn"], default="scanobjectnn")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--train-h5", default=None)
    parser.add_argument("--test-h5", default=None)
    parser.add_argument("--labels", default=None)
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
    parser.add_argument("--scanobjectnn-no-bg", action="store_true")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--model-type", choices=["pointmlp", "pointmlpelite"], default="pointmlp")
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--use-class-weights", action="store_true")
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
        default=str(TSDF_ROOT / "model" / "pointmlp"),
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
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False
        )
    elif args.dataset_type == "h5":
        if args.train_h5 is None or args.test_h5 is None or args.labels is None:
            raise ValueError("--train-h5, --test-h5, and --labels are required for --dataset-type h5")
        with open(args.labels, "r", encoding="utf-8") as handle:
            labels = [line.strip() for line in handle if line.strip()]
        train_samples = load_h5_samples(args.train_h5)
        test_samples = load_h5_samples(args.test_h5)
        train_dataset = H5ClassificationDataset(train_samples, args.num_points, labels, augment=True, seed=args.seed)
        test_dataset = H5ClassificationDataset(test_samples, args.num_points, labels, augment=False, seed=args.seed + 1)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False
        )
    else:
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
        class_weights = compute_class_weights(train_dataset, len(labels)).to(args.device)

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
    scaler = GradScaler(enabled=use_amp)

    best_acc = 0.0
    best_ckpt_path = output_dir / "pointmlp_best.pth"
    latest_ckpt_path = output_dir / "pointmlp_last.pth"
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
            with autocast(enabled=use_amp):
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
            "optimizer_state_dict": optimizer.state_dict(),
            "labels": labels,
            "num_points": args.num_points,
            "model_type": args.model_type,
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
