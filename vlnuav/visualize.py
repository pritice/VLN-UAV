import pathlib
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from .bev import BEVMap


def save_bev_images(output_dir: pathlib.Path, bev: BEVMap, masks: Optional[Dict[str, np.ndarray]] = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    occ_path = output_dir / "bev_occupancy.png"
    height_path = output_dir / "bev_height_max.png"

    plt.figure(figsize=(6, 6))
    plt.imshow(bev.occupancy, cmap="gray")
    plt.title("Occupancy")
    plt.axis("off")
    plt.savefig(occ_path, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 6))
    height = bev.height_max.copy()
    height[np.isneginf(height)] = 0
    plt.imshow(height, cmap="viridis")
    plt.title("Height Max")
    plt.axis("off")
    plt.colorbar()
    plt.savefig(height_path, bbox_inches="tight")
    plt.close()

    if bev.semantic is not None:
        plt.figure(figsize=(6, 6))
        plt.imshow(bev.semantic, cmap="tab20")
        plt.title("Semantic")
        plt.axis("off")
        plt.savefig(output_dir / "bev_semantic.png", bbox_inches="tight")
        plt.close()

    if masks:
        for name, mask in masks.items():
            plt.figure(figsize=(6, 6))
            plt.imshow(mask, cmap="magma")
            plt.title(name)
            plt.axis("off")
            plt.savefig(output_dir / f"bev_{name}.png", bbox_inches="tight")
            plt.close()
