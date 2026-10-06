from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix(matrix, output_path: Path, class_names=None):
    matrix = np.asarray(matrix)
    fig, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    axis.set_title("Breed confusion matrix")
    if class_names is None:
        class_names = [str(index) for index in range(matrix.shape[0])]
    positions = np.arange(len(class_names))
    axis.set_xticks(positions, labels=class_names, rotation=90, fontsize=6)
    axis.set_yticks(positions, labels=class_names, fontsize=6)
    axis.set_xlabel("Predicted breed")
    axis.set_ylabel("True breed")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


def save_per_breed_accuracy(accuracies, class_names, output_path: Path):
    accuracies = np.asarray(accuracies, dtype=float)
    order = np.argsort(np.nan_to_num(accuracies, nan=np.inf))
    fig, axis = plt.subplots(figsize=(9, 12))
    axis.barh(
        np.asarray(class_names)[order],
        accuracies[order],
        color="#3a7ca5",
    )
    axis.set_xlim(0, 1)
    axis.set_xlabel("Validation accuracy")
    axis.set_ylabel("Breed")
    axis.set_title("Validation accuracy by breed")
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    axis.invert_yaxis()
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
