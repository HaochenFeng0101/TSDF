import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, get_worker_info


'''
python3 detection/pointnet2/train_seg.py

python detection/pointnet2/train_seg.py \
  --use-wandb \
  --wandb-project TSDF-PointNet2-Seg \
  --wandb-run-name s3dis_pointnet2_run1

'''



REPO_ROOT = Path(__file__).resolve().parents[3]
TSDF_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSDF.detection.pointnet2.pointnet2seg import PointNet2SemSegSSG, SEG_INPUT_CHANNELS
# from TSDF.detection.training_plots import plot_segmentation_history

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    import open3d as o3d
except Exception:
    o3d = None

try:
    import wandb
except Exception:
    wandb = None


DEFAULT_DATA_ROOT = Path("data") / "S3DIS_seg"
DEFAULT_LABELS_PATH = DEFAULT_DATA_ROOT / "labels.txt"
DEFAULT_OUTPUT_DIR = Path("seg_model") / "pointnet2"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_repo_path(path_str):
    path = Path(path_str)
    if not path.is_absolute():
        path = TSDF_ROOT / path
    return path


def normalize_xyz(points):
    xyz = points[:, :3].astype(np.float32)
    centroid = xyz.mean(axis=0, keepdims=True)
    xyz = xyz - centroid
    scale = np.linalg.norm(xyz, axis=1).max()
    if scale > 0:
        xyz = xyz / scale
    points = points.astype(np.float32).copy()
    points[:, :3] = xyz
    return points


def select_input_features(points):
    if points.shape[1] < 3:
        raise ValueError(f"Expected at least xyz coordinates, got shape {points.shape}")

    if points.shape[1] < SEG_INPUT_CHANNELS:
        raise ValueError(
            f"Expected segmentation points with at least {SEG_INPUT_CHANNELS} channels "
            f"(xyz + rgb + room_coords), got shape {points.shape}"
        )

    features = points[:, :SEG_INPUT_CHANNELS].astype(np.float32).copy()
    colors = features[:, 3:6]
    if colors.size > 0 and colors.max() > 1.0:
        colors = colors / 255.0
    features[:, 3:6] = np.clip(colors, 0.0, 1.0)
    return features


def load_seg_sample(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path)
        points = data["points"]
        labels = data["labels"]
    elif suffix == ".npy":
        payload = np.load(path, allow_pickle=True)
        if isinstance(payload, np.ndarray) and payload.dtype == object and len(payload) == 1:
            payload = payload.item()
        if isinstance(payload, dict):
            points = payload["points"]
            labels = payload["labels"]
        else:
            raise ValueError(f"{path} must be a dict-like npy containing points/labels.")
    elif suffix in {".pcd", ".ply"}:
        if o3d is None:
            raise RuntimeError("open3d is required to read .pcd/.ply files.")
        cloud = o3d.io.read_point_cloud(str(path))
        if cloud.is_empty():
            raise ValueError(f"Empty point cloud: {path}")
        label_path = path.with_name(f"{path.stem}_labels.npy")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file: {label_path}")
        points = np.asarray(cloud.points, dtype=np.float32)
        if cloud.has_colors():
            colors = np.asarray(cloud.colors, dtype=np.float32)
            points = np.concatenate([points, colors], axis=1)
        labels = np.load(label_path).astype(np.int64)
    else:
        raise ValueError(f"Unsupported segmentation sample format: {path}")

    points = np.asarray(points, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"{path} points must be Nx3 or wider, got {points.shape}")
    if len(points) != len(labels):
        raise ValueError(f"{path} points/labels length mismatch: {len(points)} vs {len(labels)}")
    return points, labels


class SceneSegDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        num_points=4096,
        num_classes=13,
        augment=False,
        seed=0,
        rare_class_sampling_ratio=0.25,
        rare_class_weight_power=0.5,
    ):
        self.root = Path(root)
        self.split = split
        self.num_points = num_points
        self.num_classes = num_classes
        self.augment = augment
        self.seed = int(seed)
        self.rare_class_sampling_ratio = float(rare_class_sampling_ratio) if augment else 0.0
        self.rare_class_weight_power = float(rare_class_weight_power)
        self.global_label_counts = None
        self.class_sampling_weights = None
        self.file_sampling_weights = None
        self.sampling_cache_path = None

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.files = sorted(
            path for path in split_dir.iterdir() if path.suffix.lower() in {".npz", ".npy", ".pcd", ".ply"}
        )
        if not self.files:
            raise RuntimeError(f"No usable samples found under {split_dir}.")
        self.sampling_cache_path = (
            self.root
            / f".sampling_stats_{self.split}_c{self.num_classes}_rw{self.rare_class_weight_power:.3f}.npz"
        )
        if self.augment:
            self._build_sampling_statistics()

    def __len__(self):
        return len(self.files)

    def _make_rng(self, idx):
        if self.augment:
            worker_info = get_worker_info()
            worker_seed = worker_info.seed if worker_info is not None else self.seed
            return np.random.default_rng(worker_seed + idx)
        return np.random.default_rng(self.seed + idx)

    def _build_sampling_statistics(self):
        cached = self._load_sampling_statistics_cache()
        if cached is not None:
            print(f"Using cached sampling stats: {self.sampling_cache_path}")
            file_histograms, global_counts = cached
        else:
            print(
                f"Building sampling stats for {self.split} split over {len(self.files)} files. "
                f"This can take a while the first time."
            )
            file_histograms = []
            global_counts = np.zeros(self.num_classes, dtype=np.float64)
            iterator = self.files
            if tqdm is not None:
                iterator = tqdm(
                    self.files,
                    desc=f"sampling_stats_{self.split}",
                    leave=False,
                    dynamic_ncols=True,
                )
            for path in iterator:
                _, labels = load_seg_sample(path)
                hist = np.bincount(labels.astype(np.int64), minlength=self.num_classes).astype(np.float64)
                file_histograms.append(hist)
                global_counts += hist
            file_histograms = np.asarray(file_histograms, dtype=np.float64)
            self._save_sampling_statistics_cache(file_histograms, global_counts)
            if self.sampling_cache_path is not None:
                print(f"Saved sampling stats cache: {self.sampling_cache_path}")

        safe_counts = global_counts.copy()
        safe_counts[safe_counts == 0] = 1.0
        class_weights = np.power(safe_counts.sum() / safe_counts, self.rare_class_weight_power)
        class_weights = class_weights / np.maximum(class_weights.mean(), 1e-6)

        file_scores = []
        for hist in file_histograms:
            total = np.maximum(hist.sum(), 1.0)
            file_scores.append(float(np.dot(hist, class_weights) / total))
        file_scores = np.asarray(file_scores, dtype=np.float64)
        file_scores = file_scores / np.maximum(file_scores.mean(), 1e-6)

        self.global_label_counts = global_counts
        self.class_sampling_weights = class_weights.astype(np.float32)
        self.file_sampling_weights = torch.as_tensor(file_scores, dtype=torch.double)

    def _load_sampling_statistics_cache(self):
        if self.sampling_cache_path is None or not self.sampling_cache_path.exists():
            return None
        try:
            cache = np.load(self.sampling_cache_path, allow_pickle=False)
            cached_paths = cache["paths"]
            file_histograms = cache["file_histograms"].astype(np.float64)
            global_counts = cache["global_counts"].astype(np.float64)
        except Exception:
            return None

        current_paths = np.asarray([str(path.resolve()) for path in self.files])
        if len(cached_paths) != len(current_paths):
            return None
        if not np.array_equal(cached_paths, current_paths):
            return None
        if file_histograms.shape != (len(self.files), self.num_classes):
            return None
        return file_histograms, global_counts

    def _save_sampling_statistics_cache(self, file_histograms, global_counts):
        if self.sampling_cache_path is None:
            return
        try:
            np.savez_compressed(
                self.sampling_cache_path,
                paths=np.asarray([str(path.resolve()) for path in self.files]),
                file_histograms=np.asarray(file_histograms, dtype=np.float32),
                global_counts=np.asarray(global_counts, dtype=np.float64),
            )
        except Exception:
            return

    def _sample_point_indices(self, labels, rng):
        total_points = len(labels)
        if total_points >= self.num_points:
            uniform_choice = rng.choice(total_points, self.num_points, replace=False)
        else:
            uniform_choice = rng.choice(total_points, self.num_points, replace=True)

        if (
            not self.augment
            or self.rare_class_sampling_ratio <= 0.0
            or self.class_sampling_weights is None
            or self.num_points <= 1
        ):
            return uniform_choice.astype(np.int64)

        rare_sample_count = int(round(self.num_points * self.rare_class_sampling_ratio))
        rare_sample_count = max(0, min(self.num_points, rare_sample_count))
        if rare_sample_count == 0:
            return uniform_choice.astype(np.int64)

        point_weights = self.class_sampling_weights[labels]
        point_weight_sum = float(point_weights.sum())
        if point_weight_sum <= 0:
            return uniform_choice.astype(np.int64)
        point_weights = point_weights / point_weight_sum

        base_count = self.num_points - rare_sample_count
        if base_count > 0:
            base_choice = uniform_choice[:base_count]
        else:
            base_choice = np.empty((0,), dtype=np.int64)

        rare_choice = rng.choice(
            total_points,
            rare_sample_count,
            replace=total_points < rare_sample_count,
            p=point_weights,
        ).astype(np.int64)
        choice = np.concatenate([base_choice, rare_choice], axis=0)
        rng.shuffle(choice)
        return choice

    def __getitem__(self, idx):
        rng = self._make_rng(idx)
        points, labels = load_seg_sample(self.files[idx])
        points = select_input_features(points)

        points = normalize_xyz(points)

        choice = self._sample_point_indices(labels, rng)

        points = points[choice]
        labels = labels[choice]

        if self.augment:
            theta = rng.uniform(0.0, 2.0 * np.pi)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            rotation = np.array(
                [[cos_theta, 0.0, sin_theta], [0.0, 1.0, 0.0], [-sin_theta, 0.0, cos_theta]],
                dtype=np.float32,
            )
            points[:, :3] = points[:, :3] @ rotation.T

        return torch.from_numpy(points.T.astype(np.float32)), torch.from_numpy(labels.astype(np.int64))


def load_labels(labels_path, num_classes):
    if labels_path is None:
        return [f"class_{i}" for i in range(num_classes)]
    with open(labels_path, "r", encoding="utf-8") as handle:
        labels = [line.strip() for line in handle if line.strip()]
    if len(labels) < num_classes:
        labels.extend(f"class_{i}" for i in range(len(labels), num_classes))
    return labels[:num_classes]


def compute_class_weights(dataset, num_classes, sample_cap=200):
    counts = getattr(dataset, "global_label_counts", None)
    if counts is None:
        counts = np.zeros(num_classes, dtype=np.float64)
        limit = min(len(dataset), sample_cap)
        for idx in range(limit):
            _, labels = dataset[idx]
            counts += np.bincount(labels.numpy(), minlength=num_classes)
    else:
        counts = counts.astype(np.float64).copy()
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


def lovasz_grad(gt_sorted):
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def flatten_probas(probas, labels, ignore_index):
    if probas.ndim != 3:
        raise ValueError(f"Expected [B, C, N] probabilities, got {tuple(probas.shape)}")
    probas = probas.permute(0, 2, 1).reshape(-1, probas.shape[1])
    labels = labels.reshape(-1)
    if ignore_index is None:
        return probas, labels
    valid = labels != ignore_index
    return probas[valid], labels[valid]


def lovasz_softmax_flat(probas, labels, classes="present"):
    if probas.numel() == 0:
        return probas.sum() * 0.0

    num_classes = probas.shape[1]
    losses = []
    class_indices = range(num_classes) if classes in {"all", "present"} else classes

    for class_idx in class_indices:
        foreground = (labels == class_idx).float()
        if classes == "present" and foreground.sum() == 0:
            continue
        class_pred = probas[:, class_idx]
        errors = (foreground - class_pred).abs()
        errors_sorted, permutation = torch.sort(errors, descending=True)
        fg_sorted = foreground[permutation]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))

    if not losses:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


def lovasz_softmax_loss(logits, labels, ignore_index=-100):
    probas = F.softmax(logits, dim=1)
    probas, labels = flatten_probas(probas, labels, ignore_index)
    return lovasz_softmax_flat(probas, labels, classes="present")


def compute_seg_loss(logits, labels, class_weights=None, ignore_index=-100, lovasz_weight=0.5):
    ce_loss = F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        ignore_index=ignore_index,
    )
    if lovasz_weight <= 0:
        return ce_loss
    lovasz_loss = lovasz_softmax_loss(logits, labels, ignore_index=ignore_index)
    return (1.0 - lovasz_weight) * ce_loss + lovasz_weight * lovasz_loss


def format_per_class_iou(label_names, per_class_iou):
    parts = []
    for class_idx, label_name in enumerate(label_names):
        iou = per_class_iou[class_idx]
        value = "n/a" if iou is None else f"{iou:.3f}"
        parts.append(f"{label_name}={value}")
    return " | ".join(parts)


def evaluate(model, dataloader, device, num_classes, ignore_index=-100, class_weights=None, lovasz_weight=0.5):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    total_iou_inter = np.zeros(num_classes, dtype=np.float64)
    total_iou_union = np.zeros(num_classes, dtype=np.float64)

    iterator = dataloader
    if tqdm is not None:
        iterator = tqdm(dataloader, desc="val", leave=False, dynamic_ncols=True)

    with torch.no_grad():
        for points, labels in iterator:
            points = points.to(device)
            labels = labels.to(device)
            logits = model(points)
            loss = compute_seg_loss(
                logits,
                labels,
                class_weights=class_weights,
                ignore_index=ignore_index,
                lovasz_weight=lovasz_weight,
            )
            preds = logits.argmax(dim=1)
            valid = labels != ignore_index

            total_loss += loss.item() * valid.sum().item()
            total_correct += ((preds == labels) & valid).sum().item()
            total_seen += valid.sum().item()

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            valid_np = valid.cpu().numpy()
            for class_idx in range(num_classes):
                pred_mask = (preds_np == class_idx) & valid_np
                label_mask = (labels_np == class_idx) & valid_np
                total_iou_inter[class_idx] += np.logical_and(pred_mask, label_mask).sum()
                total_iou_union[class_idx] += np.logical_or(pred_mask, label_mask).sum()

    iou = total_iou_inter / np.maximum(total_iou_union, 1.0)
    miou = float(iou.mean())
    per_class_iou = [
        (float(iou[class_idx]) if total_iou_union[class_idx] > 0 else None)
        for class_idx in range(num_classes)
    ]
    return (
        total_loss / max(total_seen, 1),
        total_correct / max(total_seen, 1),
        miou,
        per_class_iou,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train a PointNet++ semantic segmentation model."
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
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument(
        "--rare-class-sampling-ratio",
        type=float,
        default=0.20,
        help="Fraction of sampled training points to draw with rare-class-aware weighting.",
    )
    parser.add_argument(
        "--rare-class-weight-power",
        type=float,
        default=0.40,
        help="Exponent applied to inverse class frequency for rare-class-aware sampling.",
    )
    parser.add_argument(
        "--lovasz-weight",
        type=float,
        default=0.5,
        help="Blend weight between cross-entropy and Lovasz-Softmax loss.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Model output directory. Defaults to seg_model/pointnet2 under the repository root.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="TSDF-PointNet2-Seg")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    data_root = resolve_repo_path(args.data_root)
    labels_path = resolve_repo_path(args.labels) if args.labels is not None else None
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_wandb and wandb is None:
        raise RuntimeError("wandb is not installed in this environment. Install it or omit --use-wandb.")

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

    input_channels = SEG_INPUT_CHANNELS
    labels = load_labels(labels_path, args.num_classes)
    with open(output_dir / "labels.txt", "w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(f"{label}\n")

    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    print(f"train_samples={len(train_dataset)} | val_samples={len(val_dataset)} | num_classes={args.num_classes}")
    print(
        f"device={args.device} | workers={args.workers} | batch_size={args.batch_size} | "
        f"input_channels={input_channels}"
    )
    print(
        f"rare_class_sampling_ratio={args.rare_class_sampling_ratio:.2f} | "
        f"rare_class_weight_power={args.rare_class_weight_power:.2f} | "
        f"lovasz_weight={args.lovasz_weight:.2f}"
    )

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_dataset, args.num_classes).to(args.device)

    model = PointNet2SemSegSSG(num_classes=args.num_classes, dropout=args.dropout).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=max(args.lr * 0.01, 1e-5)
    )

    best_miou = -1.0
    history = []
    best_ckpt = output_dir / "pointnet2_seg_best.pth"
    last_ckpt = output_dir / "pointnet2_seg_last.pth"

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
            loss = compute_seg_loss(
                logits,
                labels_batch,
                class_weights=class_weights,
                ignore_index=args.ignore_index,
                lovasz_weight=args.lovasz_weight,
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
            lovasz_weight=args.lovasz_weight,
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

        if args.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_miou": val_miou,
                    "lr": optimizer.param_groups[0]["lr"],
                    "best_val_miou": max(best_miou, val_miou),
                    **{
                        f"val_iou/{label_name}": (
                            val_per_class_iou[class_idx]
                            if val_per_class_iou[class_idx] is not None
                            else float("nan")
                        )
                        for class_idx, label_name in enumerate(labels)
                    },
                }
            )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "num_points": args.num_points,
            "num_classes": args.num_classes,
            "input_channels": input_channels,
            "labels": labels,
            "model_type": "pointnet2_semseg_ssg",
            "lovasz_weight": args.lovasz_weight,
        }
        torch.save(checkpoint, last_ckpt)
        if val_miou >= best_miou:
            best_miou = val_miou
            torch.save(checkpoint, best_ckpt)

    with open(output_dir / "train_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    plot_paths = plot_segmentation_history(output_dir, history, labels, "PointNet2 Segmentation")

    print(f"Training finished. Best val_mIoU={best_miou:.4f}")
    print(f"Best checkpoint: {best_ckpt}")
    print(f"Last checkpoint: {last_ckpt}")
    print(f"Labels file: {output_dir / 'labels.txt'}")
    for plot_path in plot_paths:
        print(f"plot: {plot_path}")

    if args.use_wandb:
        wandb.summary["best_checkpoint"] = str(best_ckpt)
        wandb.summary["last_checkpoint"] = str(last_ckpt)
        wandb.summary["labels_path"] = str(output_dir / "labels.txt")
        wandb.summary["metrics_path"] = str(output_dir / "train_metrics.json")
        wandb.summary["best_val_miou"] = best_miou
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
