"""Visualization helpers."""
from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def save_costmap(costmap: np.ndarray, path: str) -> None:
    """Save a costmap visualization."""

    plt.figure(figsize=(6, 6))
    plt.imshow(costmap, cmap="magma")
    plt.colorbar(label="Cost")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_path_overlay(rgb: np.ndarray, path_points: List[Tuple[int, int]], out_path: str) -> None:
    """Overlay a path on the RGB image and save."""

    canvas = rgb.copy()
    for x, y in path_points:
        if 0 <= y < canvas.shape[0] and 0 <= x < canvas.shape[1]:
            canvas[y, x] = np.array([255, 0, 0], dtype=np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(canvas)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
