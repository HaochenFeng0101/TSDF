import argparse
import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Dispatch mixed PointMLP training to the ScanObjectNN or ModelNet40 entrypoint.",
        add_help=False,
    )
    parser.add_argument(
        "--dataset-type",
        choices=["scanobjectnn", "modelnet40"],
        default="scanobjectnn",
        help="Which dataset family to use for mixed training.",
    )
    return parser


def load_entrypoint(module_name):
    return importlib.import_module(f"TSDF.detection.pointmlp.{module_name}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args, remaining_argv = parser.parse_known_args(argv)

    if args.dataset_type == "scanobjectnn":
        load_entrypoint("train_mixed_scanobjectnn").main(remaining_argv)
    else:
        load_entrypoint("train_mixed_modelnet40").main(remaining_argv)


if __name__ == "__main__":
    main()
