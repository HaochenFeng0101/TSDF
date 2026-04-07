import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, Subset


REPO_ROOT = Path(__file__).resolve().parents[3]
TSDF_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSDF.dataset.extra_object_data import (
    get_modelnet40_with_extra_dataloaders,
    get_scanobjectnn_with_extra_dataloaders,
)
from TSDF.dataset.modelnet40_data import (
    compute_sampled_dataset_size,
    get_modelnet40_dataloaders,
    load_processed_modelnet40_train_dataset,
    sample_dataset,
)
from TSDF.dataset.scanobjectnn_data import SCANOBJECTNN_LABELS, get_scanobjectnn_dataloaders
from TSDF.detection.modelnet40c.models import MODEL_SPECS, build_model, normalize_model_name
from TSDF.detection.train_pointnet_cls import (
    H5ClassificationDataset,
    PointCloudClassificationDataset,
    build_dir_splits,
    load_h5_samples,
    set_seed,
)

try:
    import wandb
except Exception:
    wandb = None


def smooth_loss(pred, target, eps=0.2):
    n_class = pred.size(1)
    one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prob = F.log_softmax(pred, dim=1)
    return -(one_hot * log_prob).sum(dim=1).mean()


def resolve_loss_name(requested_loss, model_name):
    if requested_loss is not None:
        return requested_loss
    model_key = normalize_model_name(model_name)
    return MODEL_SPECS[model_key]["default_loss"]


def compute_loss(logits, labels, loss_name, class_weights=None, label_smoothing=0.0):
    if loss_name == "smooth":
        return smooth_loss(logits, labels)
    return F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        label_smoothing=label_smoothing,
    )


def forward_pass(model, points, labels, args, class_weights=None):
    outputs = model(points)
    aux_loss = None
    if isinstance(outputs, tuple):
        logits = outputs[0]
        if len(outputs) > 1 and outputs[1] is not None and args.model_name == "pointnet":
            from TSDF.detection.pointnet_model import feature_transform_regularizer

            aux_loss = args.feature_transform_weight * feature_transform_regularizer(outputs[1])
    else:
        logits = outputs

    loss = compute_loss(
        logits,
        labels,
        loss_name=args.loss_name,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
    )
    if aux_loss is not None:
        loss = loss + aux_loss
    return logits, loss


def evaluate(model, dataloader, device, args, class_weights=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    with torch.no_grad():
        for points, labels in dataloader:
            points = points.to(device)
            labels = labels.to(device)
            logits, loss = forward_pass(model, points, labels, args, class_weights=class_weights)
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_seen += labels.size(0)
    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


def clone_samples(samples, repeat):
    cloned = []
    for _ in range(max(repeat, 1)):
        for sample in samples:
            cloned.append(copy.deepcopy(sample))
    return cloned


def build_extra_train_dataset(labels, args):
    if args.extra_train_sample is None:
        return None
    if args.extra_train_label is None:
        raise ValueError("--extra-train-label is required when --extra-train-sample is set")
    if args.extra_train_label not in labels:
        raise ValueError(
            f"Unknown extra train label '{args.extra_train_label}'. Available labels: {labels}"
        )

    sample = {
        "path": str(Path(args.extra_train_sample).expanduser().resolve()),
        "label": args.extra_train_label,
    }
    return PointCloudClassificationDataset(
        clone_samples([sample], args.extra_train_repeat),
        args.num_points,
        labels,
        augment=True,
        seed=args.seed + 101,
    )


def maybe_inject_extra_train_samples(labels, train_dataset, args):
    extra_dataset = build_extra_train_dataset(labels, args)
    if extra_dataset is None:
        return train_dataset

    combined = ConcatDataset([train_dataset, extra_dataset])
    combined.labels = getattr(train_dataset, "labels", labels)
    combined.extra_train_samples = extra_dataset.samples
    return combined


def collect_label_indices(dataset):
    if isinstance(dataset, ConcatDataset):
        indices = []
        for subdataset in dataset.datasets:
            indices.extend(collect_label_indices(subdataset))
        return indices

    if hasattr(dataset, "label_to_idx") and hasattr(dataset, "samples"):
        return [dataset.label_to_idx[sample["label"]] for sample in dataset.samples]

    if hasattr(dataset, "samples") and dataset.samples and "label_idx" in dataset.samples[0]:
        return [int(sample["label_idx"]) for sample in dataset.samples]

    if hasattr(dataset, "labels") and dataset.labels and isinstance(dataset.labels[0], int):
        return [int(label) for label in dataset.labels]

    raise ValueError(f"Cannot infer label indices from dataset type: {type(dataset).__name__}")


def compute_class_weights_for_dataset(dataset, num_classes, device):
    raw_labels = torch.tensor(collect_label_indices(dataset), dtype=torch.long)
    counts = torch.bincount(raw_labels, minlength=num_classes).float()
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return weights.to(device)


def load_processed_modelnet40_dataset(args, labels):
    dataset, processed_format = load_processed_modelnet40_train_dataset(
        root=args.modelnet40_mild_root,
        num_points=args.num_points,
        seed=args.seed + 11,
        sample_method=args.modelnet40_sample_method,
    )
    base_labels = list(dataset.labels)
    if list(labels[: len(base_labels)]) != base_labels:
        raise ValueError("Merged labels do not preserve the ModelNet40 base label order.")
    return dataset, processed_format


def build_dataloaders(args):
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
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=False,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            drop_last=False,
        )
        train_dataset = maybe_inject_extra_train_samples(labels, train_dataset, args)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=False,
        )
        return labels, train_dataset, test_dataset, train_loader, test_loader

    if args.dataset_type == "h5":
        if args.train_h5 is None or args.test_h5 is None or args.labels is None:
            raise ValueError("--train-h5, --test-h5, and --labels are required for --dataset-type h5")
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
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=False,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            drop_last=False,
        )
        train_dataset = maybe_inject_extra_train_samples(labels, train_dataset, args)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=False,
        )
        return labels, train_dataset, test_dataset, train_loader, test_loader

    if args.dataset_type == "scanobjectnn":
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
        train_dataset = maybe_inject_extra_train_samples(labels, train_dataset, args)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=False,
        )
        return labels, train_dataset, test_dataset, train_loader, test_loader

    labels, train_dataset, test_dataset, train_loader, test_loader = get_modelnet40_with_extra_dataloaders(
        modelnet40_root=args.modelnet40_root,
        extra_object_root=args.extra_object_root,
        batch_size=args.batch_size,
        num_points=args.num_points,
        workers=args.workers,
        seed=args.seed,
        include_extra=not args.no_extra_object_data,
    )
    train_dataset = maybe_inject_extra_train_samples(labels, train_dataset, args)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False,
    )
    return labels, train_dataset, test_dataset, train_loader, test_loader


def parse_args(default_model=None):
    parser = argparse.ArgumentParser(
        description="Train an official-source point cloud classifier with the local TSDF training loop."
    )
    parser.add_argument("--model-name", default=default_model, help="Model to train.")
    parser.add_argument(
        "--dataset-type",
        choices=["dir", "h5", "scanobjectnn", "modelnet40"],
        default="modelnet40",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--train-h5", default=None)
    parser.add_argument("--test-h5", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument(
        "--scanobjectnn-root",
        default=str(TSDF_ROOT / "data" / "ScanObjectNN"),
    )
    parser.add_argument("--scanobjectnn-variant", default="pb_t50_rs")
    parser.add_argument("--scanobjectnn-no-bg", action="store_true")
    parser.add_argument(
        "--extra-object-root",
        default=str(TSDF_ROOT / "data" / "extra_object"),
        help="Optional extra object directory. Class names are inferred from folder names and included by default.",
    )
    parser.add_argument(
        "--no-extra-object-data",
        action="store_true",
        help="Disable loading extra object samples from --extra-object-root.",
    )
    parser.add_argument(
        "--modelnet40-root",
        default=str(TSDF_ROOT / "data" / "ModelNet40"),
    )
    parser.add_argument(
        "--use-processed-train-data",
        action="store_true",
        help="Append a sampled processed/mild train split to the original training set.",
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
    parser.add_argument(
        "--modelnet40-sample-method",
        choices=["surface", "vertices"],
        default="surface",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--loss-name", choices=["cross_entropy", "smooth"], default=None)
    parser.add_argument("--feature-transform-weight", type=float, default=1e-3)
    parser.add_argument("--simpleview-feat-size", type=int, default=16)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument(
        "--extra-train-sample",
        default=None,
        help="Optional point cloud file to inject into the training set only.",
    )
    parser.add_argument(
        "--extra-train-label",
        default=None,
        help="Class label for --extra-train-sample. Must already exist in the dataset labels.",
    )
    parser.add_argument(
        "--extra-train-repeat",
        type=int,
        default=1,
        help="How many times to replicate --extra-train-sample inside the training set.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="TSDF-OfficialCls")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if not args.model_name:
        raise ValueError("--model-name is required.")
    args.model_name = normalize_model_name(args.model_name)
    if args.model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model '{args.model_name}'.")
    args.loss_name = resolve_loss_name(args.loss_name, args.model_name)
    if args.output_dir is None:
        args.output_dir = str(TSDF_ROOT / "model" / args.model_name)
    return args


def main(default_model=None):
    args = parse_args(default_model=default_model)
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels, train_dataset, _, train_loader, test_loader = build_dataloaders(args)
    processed_train_dataset_full = None
    processed_train_dataset = None
    processed_dataset_format = None

    if args.use_processed_train_data:
        if args.dataset_type != "modelnet40":
            raise ValueError("--use-processed-train-data is currently supported only for --dataset-type modelnet40.")
        processed_train_dataset_full, processed_dataset_format = load_processed_modelnet40_dataset(args, labels)
        target_size = compute_sampled_dataset_size(
            dataset_size=len(processed_train_dataset_full),
            ratio_denominator=args.mild_ratio_denominator,
        )
        processed_train_dataset = sample_dataset(
            processed_train_dataset_full,
            target_size=target_size,
            seed=args.seed + 23,
        )
        train_dataset = ConcatDataset([train_dataset, processed_train_dataset])
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=False,
        )

    if args.dataset_type in {"scanobjectnn", "modelnet40"}:
        print(
            "extra_object_root="
            f"{args.extra_object_root if not args.no_extra_object_data else 'disabled'}"
        )
    print(f"train_samples={len(train_dataset)} | test_samples={len(test_loader.dataset)} | num_classes={len(labels)}")
    if args.use_processed_train_data:
        processed_format_text = f" | processed_format={processed_dataset_format}" if processed_dataset_format else ""
        print(
            f"processed_train_root={Path(args.modelnet40_mild_root).resolve()}{processed_format_text} | "
            f"processed_full_samples={len(processed_train_dataset_full)} | "
            f"processed_sampled_samples={len(processed_train_dataset)} | "
            f"processed_fraction=1/{args.mild_ratio_denominator}"
        )

    labels_path = output_dir / "labels.txt"
    with open(labels_path, "w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(f"{label}\n")

    if args.extra_train_sample is not None:
        print(
            "extra training sample injected: "
            f"path={Path(args.extra_train_sample).resolve()} "
            f"label={args.extra_train_label} "
            f"repeat={args.extra_train_repeat}"
        )

    if args.use_wandb and wandb is None:
        raise RuntimeError("wandb is not installed. Install it or omit --use-wandb.")
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights_for_dataset(train_dataset, len(labels), args.device)

    model = build_model(
        args.model_name,
        num_classes=len(labels),
        num_points=args.num_points,
        simpleview_feat_size=args.simpleview_feat_size,
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
    scaler = GradScaler(enabled=use_amp)

    best_acc = 0.0
    best_ckpt_path = output_dir / f"{args.model_name}_best.pth"
    latest_ckpt_path = output_dir / f"{args.model_name}_last.pth"
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
                logits, loss = forward_pass(model, points, labels_batch, args, class_weights=class_weights)

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
        val_loss, val_acc = evaluate(model, test_loader, args.device, args, class_weights=class_weights)
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
            "model_name": args.model_name,
            "loss_name": args.loss_name,
            "val_acc": val_acc,
            "dataset_type": args.dataset_type,
            "extra_object_root": None if args.no_extra_object_data else str(Path(args.extra_object_root).resolve()),
            "use_processed_train_data": args.use_processed_train_data,
            "mild_ratio_denominator": args.mild_ratio_denominator if args.use_processed_train_data else None,
        }
        if args.dataset_type == "modelnet40":
            ckpt["modelnet40_root"] = str(Path(args.modelnet40_root).resolve())
            if args.use_processed_train_data:
                ckpt["modelnet40_mild_root"] = str(Path(args.modelnet40_mild_root).resolve())
                ckpt["processed_dataset_format"] = processed_dataset_format
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
