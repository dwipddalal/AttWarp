from __future__ import annotations

from pathlib import Path
from typing import List


def plot_train_val_curves(
    train_values: List[float],
    val_values: List[float],
    out_path: Path,
    title: str = "Training/Validation Curves",
    ylabel: str = "Loss",
) -> None:
    """
    Save a PNG plot of train and validation curves over epochs.
    Best-effort (never raises).
    """
    try:
        import matplotlib.pyplot as plt

        epochs = list(range(1, len(train_values) + 1))
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, train_values, label="Train", marker="o")
        plt.plot(epochs, val_values, label="Val", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(str(out_path))
        plt.close()
    except Exception:
        pass


