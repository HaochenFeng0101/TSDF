from pathlib import Path

import matplotlib.pyplot as plt


def _prepare_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_classification_history(output_dir, history, title):
    output_dir = _prepare_output_dir(output_dir)

    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    train_acc = [item["train_acc"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    val_acc = [item["val_acc"] for item in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14)

    plots = [
        (axes[0, 0], train_loss, "Train Loss", "Loss", "tab:blue"),
        (axes[0, 1], train_acc, "Train Accuracy", "Accuracy", "tab:green"),
        (axes[1, 0], val_loss, "Validation Loss", "Loss", "tab:orange"),
        (axes[1, 1], val_acc, "Validation Accuracy", "Accuracy", "tab:red"),
    ]

    for ax, values, plot_title, ylabel, color in plots:
        ax.plot(epochs, values, color=color, linewidth=2)
        ax.set_title(plot_title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    output_path = output_dir / "train_metrics.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [output_path]


def plot_segmentation_history(output_dir, history, labels, title):
    output_dir = _prepare_output_dir(output_dir)

    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    train_acc = [item["train_acc"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    val_acc = [item["val_acc"] for item in history]
    val_miou = [item["val_miou"] for item in history]

    overview_fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    overview_fig.suptitle(title, fontsize=14)

    overview_plots = [
        (axes[0], train_loss, val_loss, "Loss", "Loss", "tab:blue", "tab:orange"),
        (axes[1], train_acc, val_acc, "Accuracy", "Accuracy", "tab:green", "tab:red"),
        (axes[2], val_miou, None, "Validation mIoU", "mIoU", "tab:purple", None),
    ]

    for ax, primary_values, secondary_values, plot_title, ylabel, primary_color, secondary_color in overview_plots:
        ax.plot(epochs, primary_values, color=primary_color, linewidth=2, label=plot_title if secondary_values is None else f"Train {ylabel}")
        if secondary_values is not None:
            secondary_label = "Validation Loss" if ylabel == "Loss" else "Validation Accuracy"
            ax.plot(epochs, secondary_values, color=secondary_color, linewidth=2, label=secondary_label)
            ax.legend()
        ax.set_title(plot_title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)

    overview_fig.tight_layout()
    overview_path = output_dir / "train_metrics.png"
    overview_fig.savefig(overview_path, dpi=200, bbox_inches="tight")
    plt.close(overview_fig)

    per_class_fig, per_class_ax = plt.subplots(figsize=(12, 7))
    per_class_fig.suptitle(f"{title} Per-Class IoU", fontsize=14)
    for label in labels:
        values = [
            item.get("val_per_class_iou", {}).get(label)
            for item in history
        ]
        if all(value is None for value in values):
            continue
        cleaned_values = [float("nan") if value is None else value for value in values]
        per_class_ax.plot(epochs, cleaned_values, linewidth=1.5, label=label)

    per_class_ax.set_xlabel("Epoch")
    per_class_ax.set_ylabel("IoU")
    per_class_ax.grid(True, linestyle="--", alpha=0.35)
    if labels:
        per_class_ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    per_class_fig.tight_layout()
    per_class_path = output_dir / "train_per_class_iou.png"
    per_class_fig.savefig(per_class_path, dpi=200, bbox_inches="tight")
    plt.close(per_class_fig)

    return [overview_path, per_class_path]
