from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_series(history: Iterable[Mapping[str, object]], key: str):
    values = []
    for entry in history:
        values.append(_to_float(entry.get(key)))
    return values


def _extract_epochs(history: Iterable[Mapping[str, object]]):
    epochs = []
    for index, entry in enumerate(history, start=1):
        epoch = entry.get("epoch", index)
        try:
            epochs.append(int(epoch))
        except (TypeError, ValueError):
            epochs.append(index)
    return epochs


def plot_classification_history(output_dir, history, title="Classification"):
    """Plot loss/accuracy curves from a classification training history.

    Returns a list of generated plot paths. If plotting dependencies are missing
    or history is empty, this function returns an empty list.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = list(history)
    if not history:
        return []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plot skipped: matplotlib is unavailable ({exc})")
        return []

    epochs = _extract_epochs(history)
    train_loss = _extract_series(history, "train_loss")
    val_loss = _extract_series(history, "val_loss")
    train_acc = _extract_series(history, "train_acc")
    val_acc = _extract_series(history, "val_acc")

    saved_paths = []

    def save_plot(path, ylabel, train_values, val_values, chart_title):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_values, label="train", linewidth=2)
        ax.plot(epochs, val_values, label="val", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(chart_title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved_paths.append(path)

    save_plot(
        output_dir / "classification_loss.png",
        "Loss",
        train_loss,
        val_loss,
        f"{title} - Loss",
    )
    save_plot(
        output_dir / "classification_accuracy.png",
        "Accuracy",
        train_acc,
        val_acc,
        f"{title} - Accuracy",
    )
    return saved_paths


__all__ = ["plot_classification_history"]
