import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_input_candidates(model_name: str) -> List[Path]:
    return [
        PROJECT_ROOT / "model" / model_name / "train_metrics.json",
        PROJECT_ROOT / "model" / model_name / "train-metrics.json",
        PROJECT_ROOT / "detection" / "model" / model_name / "train_metrics.json",
        PROJECT_ROOT / "detection" / "model" / model_name / "train-metrics.json",
    ]


def resolve_input_path(cli_path: Optional[str], model_name: str) -> Path:
    if cli_path:
        path = Path(cli_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Metrics file not found: {path}")
        return path

    candidates = build_input_candidates(model_name)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find a metrics file automatically for model '{model_name}'. Checked:\n"
        f"{searched}\n"
        "Please provide one with --input."
    )


def load_metrics(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    if not isinstance(metrics, list) or not metrics:
        raise ValueError("Metrics file must contain a non-empty JSON list.")

    required_keys = {"epoch", "train_loss", "train_acc", "val_loss", "val_acc"}
    for index, item in enumerate(metrics):
        if not isinstance(item, dict):
            raise ValueError(f"Entry {index} is not a JSON object.")
        missing = required_keys - item.keys()
        if missing:
            raise ValueError(f"Entry {index} is missing keys: {sorted(missing)}")

    return metrics


def plot_metrics(
    metrics: List[Dict], output_path: Optional[Path], show: bool, model_name: str
) -> None:
    epochs = [item["epoch"] for item in metrics]
    train_loss = [item["train_loss"] for item in metrics]
    train_acc = [item["train_acc"] for item in metrics]
    val_loss = [item["val_loss"] for item in metrics]
    val_acc = [item["val_acc"] for item in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("{0} Training Metrics".format(model_name), fontsize=14)

    plots = [
        (axes[0, 0], train_loss, "Train Loss", "Loss", "tab:blue"),
        (axes[0, 1], train_acc, "Train Accuracy", "Accuracy", "tab:green"),
        (axes[1, 0], val_loss, "Validation Loss", "Loss", "tab:orange"),
        (axes[1, 1], val_acc, "Validation Accuracy", "Accuracy", "tab:red"),
    ]

    for ax, values, title, ylabel, color in plots:
        ax.plot(epochs, values, color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training metrics for a specified model.")
    parser.add_argument(
        "model_name",
        nargs="?",
        default="pointmlp",
        help="Model name whose metrics live under model/<model_name>/",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to train_metrics.json or train-metrics.json. If omitted, the script searches common locations for the selected model.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the plotted figure. If omitted, it is saved to model/<model_name>/train_metrics.png.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure in a window in addition to saving it.",
    )
    args = parser.parse_args()

    input_path = resolve_input_path(args.input, args.model_name)
    metrics = load_metrics(input_path)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = (PROJECT_ROOT / "model" / args.model_name / "train_metrics.png").resolve()

    plot_metrics(metrics, output_path=output_path, show=args.show, model_name=args.model_name)


if __name__ == "__main__":
    main()
