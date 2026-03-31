import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

'''

python detection/pointmlp/train_seg.py \
  --use-class-weights \
  --rare-class-sampling-ratio 0.35 \
  --rare-class-weight-power 0.75

'''

REPO_ROOT = Path(__file__).resolve().parents[3]
TSDF_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSDF.detection.pointmlp.pointmlp_seg import PointMLPSemSeg, SEG_INPUT_CHANNELS
from TSDF.detection.pointnet2.train_seg import (
    SceneSegDataset,
    compute_class_weights,
    evaluate,
    format_per_class_iou,
    load_labels,
    resolve_repo_path,
    set_seed,
    tqdm,
)


DEFAULT_DATA_ROOT = Path("data") / "S3DIS_seg"
DEFAULT_LABELS_PATH = DEFAULT_DATA_ROOT / "labels.txt"
DEFAULT_OUTPUT_DIR = Path("seg_model") / "pointmlp"


def main():
    parser = argparse.ArgumentParser(
        description="Train a PointMLP semantic segmentation model on S3DIS."
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Dataset root. Defaults to data/S3DIS_seg under the repository root.",
    )
    parser.add_argument(
        "--labels",
        default=str(DEFAULT_LABELS_PATH),
        help="Label file, one class name per line. Defaults to data/S3DIS_seg/labels.txt.",
    )
    parser.add_argument("--num-classes", type=int, default=13)
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore-index", type=int, default=-100)
    parser.add_argument("--model-type", choices=["pointmlp", "pointmlpelite"], default="pointmlp")
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument(
        "--rare-class-sampling-ratio",
        type=float,
        default=0.25,
        help="Fraction of sampled training points to draw with rare-class-aware weighting.",
    )
    parser.add_argument(
        "--rare-class-weight-power",
        type=float,
        default=0.5,
        help="Exponent applied to inverse class frequency for rare-class-aware sampling.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Model output directory. Defaults to seg_model/pointmlp under the repository root.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    data_root = resolve_repo_path(args.data_root)
    labels_path = resolve_repo_path(args.labels) if args.labels is not None else None
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = SceneSegDataset(
        root=data_root,
        split="train",
        num_points=args.num_points,
        num_classes=args.num_classes,
        augment=True,
        seed=args.seed,
        rare_class_sampling_ratio=args.rare_class_sampling_ratio,
        rare_class_weight_power=args.rare_class_weight_power,
    )
    val_dataset = SceneSegDataset(
        root=data_root,
        split="val",
        num_points=args.num_points,
        num_classes=args.num_classes,
        augment=False,
        seed=args.seed + 1,
    )

    train_sampler = None
    if train_dataset.file_sampling_weights is not None:
        train_sampler = WeightedRandomSampler(
            train_dataset.file_sampling_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )

    labels = load_labels(labels_path, args.num_classes)
    with open(output_dir / "labels.txt", "w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(f"{label}\n")

    print(f"train_samples={len(train_dataset)} | val_samples={len(val_dataset)} | num_classes={args.num_classes}")
    print(
        f"device={args.device} | workers={args.workers} | batch_size={args.batch_size} | "
        f"input_channels={SEG_INPUT_CHANNELS} | model_type={args.model_type}"
    )
    print(
        f"rare_class_sampling_ratio={args.rare_class_sampling_ratio:.2f} | "
        f"rare_class_weight_power={args.rare_class_weight_power:.2f}"
    )

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_dataset, args.num_classes).to(args.device)

    model = PointMLPSemSeg(
        num_classes=args.num_classes,
        num_points=args.num_points,
        model_type=args.model_type,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=max(args.lr * 0.01, 1e-5)
    )

    best_miou = -1.0
    history = []
    best_ckpt = output_dir / "pointmlp_seg_best.pth"
    last_ckpt = output_dir / "pointmlp_seg_last.pth"

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
            logits = model(points)
            loss = F.cross_entropy(
                logits,
                labels_batch,
                weight=class_weights,
                ignore_index=args.ignore_index,
            )
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            valid = labels_batch != args.ignore_index
            total_loss += loss.item() * valid.sum().item()
            total_correct += ((preds == labels_batch) & valid).sum().item()
            total_seen += valid.sum().item()
            if tqdm is not None:
                iterator.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / max(total_seen, 1)
        train_acc = total_correct / max(total_seen, 1)
        val_loss, val_acc, val_miou, val_per_class_iou = evaluate(
            model,
            val_loader,
            args.device,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            class_weights=class_weights,
        )
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_miou": val_miou,
                "val_per_class_iou": {
                    label_name: val_per_class_iou[class_idx]
                    for class_idx, label_name in enumerate(labels)
                },
            }
        )

        print(
            f"epoch {epoch:03d} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | val_mIoU {val_miou:.4f}"
        )
        print(f"val_per_class_iou | {format_per_class_iou(labels, val_per_class_iou)}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "num_points": args.num_points,
            "num_classes": args.num_classes,
            "input_channels": SEG_INPUT_CHANNELS,
            "labels": labels,
            "model_type": args.model_type,
            "dropout": args.dropout,
            "task": "semantic_segmentation",
            "arch": "pointmlp_semseg",
        }
        torch.save(checkpoint, last_ckpt)
        if val_miou >= best_miou:
            best_miou = val_miou
            torch.save(checkpoint, best_ckpt)

    with open(output_dir / "train_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"Training finished. Best val_mIoU={best_miou:.4f}")
    print(f"Best checkpoint: {best_ckpt}")
    print(f"Last checkpoint: {last_ckpt}")
    print(f"Labels file: {output_dir / 'labels.txt'}")


if __name__ == "__main__":
    main()
