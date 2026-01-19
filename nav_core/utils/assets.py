"""Synthetic test asset generation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.image as mpimg
import numpy as np


def _save_rgb(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(str(path), image.astype(np.uint8))


def generate_test_assets(outdir: str | Path, size: Tuple[int, int] = (128, 128)) -> Dict[str, Path]:
    """Generate deterministic synthetic test images.

    Args:
        outdir: Directory to place generated assets.
        size: (height, width) of the generated images.

    Returns:
        Mapping from asset name to file path.
    """

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    h, w = size

    assets: Dict[str, Path] = {}

    # t1_water: horizontal blue band in the middle.
    img = np.full((h, w, 3), 220, dtype=np.uint8)
    band_h = h // 5
    start = h // 2 - band_h // 2
    img[start : start + band_h, :, :] = np.array([40, 80, 200], dtype=np.uint8)
    path = out_path / "t1_water.png"
    _save_rgb(img, path)
    assets["t1_water.png"] = path

    # t2_road: gray road from bottom to top with small obstacles.
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    road_w = w // 4
    road_start = w // 2 - road_w // 2
    img[:, road_start : road_start + road_w, :] = np.array([130, 130, 130], dtype=np.uint8)
    img[h // 3 : h // 3 + 5, road_start + 5 : road_start + 10] = np.array(
        [20, 20, 20], dtype=np.uint8
    )
    img[h // 2 : h // 2 + 5, road_start + road_w - 15 : road_start + road_w - 5] = np.array(
        [20, 20, 20], dtype=np.uint8
    )
    path = out_path / "t2_road.png"
    _save_rgb(img, path)
    assets["t2_road.png"] = path

    # t3_wall: vertical black wall with a narrow gap.
    img = np.full((h, w, 3), 210, dtype=np.uint8)
    wall_w = w // 10
    wall_x = w // 2 - wall_w // 2
    img[:, wall_x : wall_x + wall_w, :] = np.array([10, 10, 10], dtype=np.uint8)
    gap_h = h // 8
    gap_y = h // 2 - gap_h // 2
    img[gap_y : gap_y + gap_h, wall_x : wall_x + wall_w, :] = np.array(
        [210, 210, 210], dtype=np.uint8
    )
    path = out_path / "t3_wall.png"
    _save_rgb(img, path)
    assets["t3_wall.png"] = path

    # t7_nopath: solid obstacle wall blocking path.
    img = np.full((h, w, 3), 220, dtype=np.uint8)
    block_h = h // 6
    block_y = h // 2 - block_h // 2
    img[block_y : block_y + block_h, :, :] = np.array([0, 0, 0], dtype=np.uint8)
    path = out_path / "t7_nopath.png"
    _save_rgb(img, path)
    assets["t7_nopath.png"] = path

    return assets
