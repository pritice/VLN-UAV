"""Placeholder mapping utilities for grid layers.

TODO: replace with SAM3, depth estimation, or 3D fusion modules.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from nav_core.contracts import GridLayers

DEFAULT_SEMANTIC_LABELS = [
    "dark",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
    "white",
]


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(
        np.float32
    )


def generate_layers(rgb: np.ndarray) -> GridLayers:
    """Generate placeholder grid layers from a single RGB frame.

    Args:
        rgb: HxWx3 uint8 RGB image.

    Returns:
        GridLayers with semantic_id, obstacle, and uncertainty.
    """

    gray = _to_gray(rgb)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    grad = gx + gy

    threshold = float(grad.mean() + grad.std())
    obstacle = (grad > threshold).astype(np.uint8)

    semantic_id = (
        (rgb[..., 0] > 128).astype(np.int32)
        + (rgb[..., 1] > 128).astype(np.int32) * 2
        + (rgb[..., 2] > 128).astype(np.int32) * 4
    )

    uncertainty = grad / (grad.max() + 1e-6)
    uncertainty = uncertainty.astype(np.float32)

    return GridLayers(
        semantic_id=semantic_id,
        obstacle=obstacle,
        uncertainty=uncertainty,
    )


def semantic_label_from_id(semantic_id: int) -> str:
    """Map a semantic ID to a coarse label."""

    if 0 <= semantic_id < len(DEFAULT_SEMANTIC_LABELS):
        return DEFAULT_SEMANTIC_LABELS[semantic_id]
    return "unknown"


def semantic_id_from_label(label: str) -> Tuple[int, ...]:
    """Return semantic IDs that match a coarse label or alias."""

    if label in DEFAULT_SEMANTIC_LABELS:
        return (DEFAULT_SEMANTIC_LABELS.index(label),)

    alias = {
        "water": ("blue", "cyan"),
        "road": ("dark",),
        "vegetation": ("green",),
    }
    if label in alias:
        return tuple(DEFAULT_SEMANTIC_LABELS.index(name) for name in alias[label])
    return tuple()
