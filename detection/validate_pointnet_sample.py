import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

# python detection/validate_pointnet_sample.py \
#   --checkpoint model/pointnet_best.pth \
#   --scanobjectnn-root data/ScanObjectNN \
#   --scanobjectnn-variant pb_t50_rs \
#   --use-all-points



REPO_ROOT = Path(__file__).resolve().parents[2]
TSDF_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSDF.dataset.scanobjectnn_data import SCANOBJECTNN_LABELS, ScanObjectNNDataset
from TSDF.detection.pointnet_model import PointNetCls


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        raise RuntimeError(f"Missing checkpoint keys: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    return checkpoint


def sample_points(points, num_points, seed):
    rng = np.random.default_rng(seed)
    if len(points) >= num_points:
        indices = rng.choice(len(points), num_points, replace=False)
    else:
        indices = rng.choice(len(points), num_points, replace=True)
    sampled = points[indices].astype(np.float32)
    centroid = sampled.mean(axis=0, keepdims=True)
    sampled = sampled - centroid
    scale = np.linalg.norm(sampled, axis=1).max()
    if scale > 0:
        sampled = sampled / scale
    return sampled


def prepare_points(points, num_points, seed, use_all_points=False):
    if use_all_points:
        sampled = points.astype(np.float32)
        centroid = sampled.mean(axis=0, keepdims=True)
        sampled = sampled - centroid
        scale = np.linalg.norm(sampled, axis=1).max()
        if scale > 0:
            sampled = sampled / scale
        return sampled
    return sample_points(points, num_points, seed)


def predict_with_votes(model, points, device, num_points, num_votes, use_all_points=False):
    logits_votes = []
    for vote_idx in range(max(num_votes, 1)):
        prepared = prepare_points(
            points,
            num_points=num_points,
            seed=vote_idx,
            use_all_points=use_all_points,
        )
        tensor = torch.from_numpy(prepared.T).unsqueeze(0).to(device=device)
        with torch.no_grad():
            logits, _ = model(tensor)
        logits_votes.append(logits[0].detach().cpu())

    mean_logits = torch.stack(logits_votes, dim=0).mean(dim=0)
    probs = torch.softmax(mean_logits, dim=0)
    pred = int(torch.argmax(probs).item())
    return pred, probs


def evaluate_dataset(model, dataset, device, labels, num_points, num_votes, use_all_points):
    model.eval()
    total_correct = 0
    class_correct = np.zeros(len(labels), dtype=np.int64)
    class_total = np.zeros(len(labels), dtype=np.int64)

    for idx in range(len(dataset)):
        raw_points = dataset.data[idx][:, :3]
        target = int(dataset.labels[idx])
        pred, _ = predict_with_votes(
            model,
            raw_points,
            device,
            num_points=num_points,
            num_votes=num_votes,
            use_all_points=use_all_points,
        )
        total_correct += int(pred == target)
        class_correct[target] += int(pred == target)
        class_total[target] += 1

    overall_acc = total_correct / max(len(dataset), 1)
    class_acc = class_correct / np.maximum(class_total, 1)
    mean_class_acc = float(class_acc.mean())
    return overall_acc, mean_class_acc, class_acc


def inspect_one_sample(
    model,
    dataset,
    labels,
    num_points,
    device,
    index,
    num_votes,
    use_all_points=False,
    topk=3,
):
    raw_points = dataset.data[index][:, :3]
    target = int(dataset.labels[index])
    pred, probs = predict_with_votes(
        model,
        raw_points,
        device,
        num_points=num_points,
        num_votes=num_votes,
        use_all_points=use_all_points,
    )

    topk = min(topk, len(labels))
    top_probs, top_indices = torch.topk(probs, k=topk)

    print(f"sample_index: {index}")
    print(f"raw_num_points: {len(raw_points)}")
    print(f"evaluation_mode: {'all_points' if use_all_points else f'{num_votes}_vote_sampling'}")
    print(f"ground_truth: {labels[target]}")
    print(f"predicted: {labels[pred]}")
    print(f"confidence: {float(top_probs[0]):.4f}")
    print("top_predictions:")
    for rank in range(topk):
        cls_idx = int(top_indices[rank].item())
        score = float(top_probs[rank].item())
        print(f"  {rank + 1}. {labels[cls_idx]} ({score:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PointNet on ScanObjectNN and inspect a single test sample."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(TSDF_ROOT / "model" / "pointnet" / "pointnet_best.pth"),
        help="Checkpoint path.",
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
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument(
        "--num-votes",
        type=int,
        default=1,
        help="Number of resampled votes per object during evaluation.",
    )
    parser.add_argument(
        "--use-all-points",
        action="store_true",
        help="Evaluate each object with its full point cloud instead of fixed-size sampling.",
    )
    parser.add_argument("--index", type=int, default=None, help="Inspect this sample index.")
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Only evaluate the chosen sample and skip the full-dataset pass.",
    )
    parser.add_argument(
        "--run-full-eval",
        action="store_true",
        help="Run the full dataset evaluation before inspecting one sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed. Leave unset for a different random sample each run.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    labels = SCANOBJECTNN_LABELS
    model = PointNetCls(k=len(labels)).to(args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, args.device)
    model.eval()

    num_points = args.num_points or checkpoint.get("num_points", 1024)
    dataset = ScanObjectNNDataset(
        root=args.scanobjectnn_root,
        split=args.split,
        variant=args.scanobjectnn_variant,
        num_points=num_points,
        use_background=not args.scanobjectnn_no_bg,
        normalize=True,
        augment=False,
        seed=args.seed,
    )

    if args.index is None:
        args.index = random.randrange(len(dataset))
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"index {args.index} out of range for dataset of size {len(dataset)}")

    should_run_full_eval = args.run_full_eval and not args.sample_only
    if should_run_full_eval:
        overall_acc, mean_class_acc, class_acc = evaluate_dataset(
            model,
            dataset,
            args.device,
            labels,
            num_points,
            num_votes=args.num_votes,
            use_all_points=args.use_all_points,
        )
        print(f"overall_accuracy: {overall_acc:.4f}")
        print(f"mean_class_accuracy: {mean_class_acc:.4f}")

        worst_idx = int(np.argmin(class_acc))
        best_idx = int(np.argmax(class_acc))
        print(f"best_class: {labels[best_idx]} ({class_acc[best_idx]:.4f})")
        print(f"worst_class: {labels[worst_idx]} ({class_acc[worst_idx]:.4f})")

    inspect_one_sample(
        model,
        dataset,
        labels,
        num_points,
        args.device,
        index=args.index,
        num_votes=args.num_votes,
        use_all_points=args.use_all_points,
    )


if __name__ == "__main__":
    main()
