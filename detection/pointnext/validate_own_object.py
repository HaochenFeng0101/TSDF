import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSDF.detection.pointnext.validate import main as validate_main


def main():
    parser = argparse.ArgumentParser(
        description="Validate PointNeXt on a custom point cloud or a dataset sample."
    )
    parser.add_argument("pcd", nargs="?", default=None, help="Optional input point cloud path.")
    parser.add_argument(
        "--dataset",
        choices=["scanobjectnn", "modelnet40"],
        default=None,
        help="Validate on an internal dataset sample instead of a custom point cloud.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--scanobjectnn-root", default=None)
    parser.add_argument("--scanobjectnn-variant", default=None)
    parser.add_argument("--scanobjectnn-no-bg", action="store_true")
    parser.add_argument("--modelnet40-root", default=None)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--num-votes", type=int, default=1)
    parser.add_argument("--use-all-points", action="store_true")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--visualize-raw-points", action="store_true")
    args = parser.parse_args()

    if args.dataset is None and args.pcd is None:
        parser.error("Provide either a point cloud path or --dataset.")

    forwarded = []
    if args.pcd is not None:
        forwarded.extend(["--point-cloud", args.pcd])
    if args.dataset is not None:
        forwarded.extend(["--dataset-type", args.dataset])

    if args.checkpoint is not None:
        forwarded.extend(["--checkpoint", args.checkpoint])
    if args.labels is not None:
        forwarded.extend(["--labels", args.labels])
    if args.scanobjectnn_root is not None:
        forwarded.extend(["--scanobjectnn-root", args.scanobjectnn_root])
    if args.scanobjectnn_variant is not None:
        forwarded.extend(["--scanobjectnn-variant", args.scanobjectnn_variant])
    if args.scanobjectnn_no_bg:
        forwarded.append("--scanobjectnn-no-bg")
    if args.modelnet40_root is not None:
        forwarded.extend(["--modelnet40-root", args.modelnet40_root])

    forwarded.extend(["--split", args.split])
    forwarded.extend(["--num-votes", str(args.num_votes)])
    forwarded.extend(["--topk", str(args.topk)])

    if args.num_points is not None:
        forwarded.extend(["--num-points", str(args.num_points)])
    if args.index is not None:
        forwarded.extend(["--index", str(args.index)])
    if args.seed is not None:
        forwarded.extend(["--seed", str(args.seed)])
    if args.device is not None:
        forwarded.extend(["--device", args.device])

    if args.use_all_points:
        forwarded.append("--use-all-points")
    if args.visualize_raw_points:
        forwarded.append("--visualize-raw-points")
    if not args.no_visualize:
        forwarded.append("--visualize")

    validate_main(forwarded)


if __name__ == "__main__":
    main()
