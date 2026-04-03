import argparse
import random
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
TSDF_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSDF.dataset.scanobjectnn_data import ScanObjectNNDataset
from TSDF.detection.pointnet2.pointnet2 import PointNet2ClsSSG
from TSDF.detection.pointnet2.validate import (
    inspect_one_sample,
    inspect_point_cloud,
    load_checkpoint,
    load_labels,
    load_point_cloud_points,
    set_seed,
    visualize_point_cloud,
)


def main():
    parser = argparse.ArgumentParser(
        description="Validate PointNet++ on a custom point cloud or on a ScanObjectNN sample."
    )
    parser.add_argument("pcd", nargs="?", default=None, help="Optional input point cloud path (.pcd or .ply).")
    parser.add_argument(
        "--dataset",
        choices=["scanobjectnn"],
        default=None,
        help="Validate on a built-in dataset sample instead of a custom point cloud.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(TSDF_ROOT / "model" / "pointnet2" / "pointnet2_best.pth"),
        help="PointNet++ checkpoint path.",
    )
    parser.add_argument(
        "--labels",
        default=str(TSDF_ROOT / "model" / "pointnet2" / "labels.txt"),
        help="Label file.",
    )
    parser.add_argument(
        "--scanobjectnn-root",
        default=str(TSDF_ROOT / "data" / "ScanObjectNN"),
        help="ScanObjectNN root directory.",
    )
    parser.add_argument("--scanobjectnn-variant", default="pb_t50_rs")
    parser.add_argument("--scanobjectnn-no-bg", action="store_true")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--index", type=int, default=None, help="Dataset sample index. Defaults to a random sample.")
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--num-votes", type=int, default=1)
    parser.add_argument("--use-all-points", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--visualize-raw-points", action="store_true")
    args = parser.parse_args()

    if args.dataset is None and args.pcd is None:
        parser.error("Provide either a point cloud path or --dataset scanobjectnn.")

    set_seed(args.seed)
    labels = load_labels(args.labels if args.dataset is None else None)

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    except Exception:
        ckpt = None

    ckpt_num_points = ckpt.get("num_points", 1024) if isinstance(ckpt, dict) else 1024
    num_points = args.num_points or ckpt_num_points
    model = PointNet2ClsSSG(num_classes=len(labels)).to(args.device)
    load_checkpoint(model, args.checkpoint, args.device)
    model.eval()

    if args.dataset == "scanobjectnn":
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

        result = inspect_one_sample(
            model=model,
            dataset=dataset,
            labels=labels,
            num_points=num_points,
            device=args.device,
            index=args.index,
            num_votes=args.num_votes,
            use_all_points=args.use_all_points,
        )
    else:
        points = load_point_cloud_points(args.pcd)
        print(f"checkpoint: {Path(args.checkpoint).resolve()}")
        print(f"point_cloud: {Path(args.pcd).resolve()}")
        result = inspect_point_cloud(
            model=model,
            points=points,
            labels=labels,
            num_points=num_points,
            device=args.device,
            num_votes=args.num_votes,
            use_all_points=args.use_all_points,
        )

    if not args.no_visualize:
        points_to_show = result["raw_points"] if args.visualize_raw_points else result["prepared_points"]
        title = f"PointNet++ | pred={result['pred_label']}"
        if "target_label" in result:
            title = f"PointNet++ | gt={result['target_label']} | pred={result['pred_label']}"
        visualize_point_cloud(points_to_show, title)


if __name__ == "__main__":
    main()
