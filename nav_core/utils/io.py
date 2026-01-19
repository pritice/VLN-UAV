"""I/O helpers for reading images and building observations."""
from __future__ import annotations

from typing import Union

import matplotlib.image as mpimg
import numpy as np

from nav_core.contracts import Observation


def load_rgb(image: Union[str, np.ndarray]) -> np.ndarray:
    """Load an RGB image from a path or return a copy of the array.

    Args:
        image: Path to image file or ndarray HxWx3.

    Returns:
        RGB image as uint8 array.
    """

    if isinstance(image, str):
        data = mpimg.imread(image)
    else:
        data = np.array(image)

    if data.dtype != np.uint8:
        data = (np.clip(data, 0.0, 1.0) * 255.0).astype(np.uint8)

    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError("Expected an HxWx3 RGB image")

    return data


def build_observation(image: Union[str, np.ndarray]) -> Observation:
    """Create an Observation from image input."""

    rgb = load_rgb(image)
    return Observation(rgb=rgb)
