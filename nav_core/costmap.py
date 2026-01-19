"""Costmap generation logic."""
from __future__ import annotations

import importlib.util
from typing import Dict

import numpy as np

from nav_core.contracts import Costmap, CostmapMeta, GridLayers, InstructionDSL
from nav_core import mapping


def _distance_transform(obstacle: np.ndarray) -> np.ndarray:
    """Compute distance to nearest obstacle with optional SciPy support."""

    if importlib.util.find_spec("scipy") is not None:
        from scipy.ndimage import distance_transform_edt

        return distance_transform_edt(1 - obstacle)
    return _approx_distance_transform(obstacle)


def _approx_distance_transform(obstacle: np.ndarray, max_radius: int = 8) -> np.ndarray:
    """Approximate distance transform with iterative dilation."""

    h, w = obstacle.shape
    distance = np.full((h, w), max_radius, dtype=np.float32)
    current = obstacle.astype(bool)
    distance[current] = 0.0

    for r in range(1, max_radius + 1):
        padded = np.pad(current, 1, mode="edge")
        neighbors = (
            padded[1:-1, :-2]
            | padded[1:-1, 2:]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[:-2, :-2]
            | padded[:-2, 2:]
            | padded[2:, :-2]
            | padded[2:, 2:]
        )
        newly_reached = neighbors & ~current
        distance[newly_reached] = np.minimum(distance[newly_reached], r)
        current = current | neighbors

    return distance


def build_costmap(
    layers: GridLayers,
    dsl: InstructionDSL,
    resolution: float = 0.1,
    origin: tuple = (0.0, 0.0),
    frame_id: str = "map",
    uncertainty_weight: float = 2.0,
) -> Costmap:
    """Build a costmap from grid layers and DSL.

    Args:
        layers: GridLayers with obstacle, semantic, and uncertainty.
        dsl: Parsed instruction DSL.
        resolution: Grid resolution in meters.
        origin: Origin of the costmap frame.
        frame_id: Frame identifier.
        uncertainty_weight: Weight applied to the uncertainty layer.

    Returns:
        Costmap with computed grid costs.
    """

    obstacle_mask = layers.obstacle.astype(bool)
    grid = np.ones_like(layers.uncertainty, dtype=np.float32)

    obstacle_cost = 1e6
    grid[obstacle_mask] = obstacle_cost

    safety_radius = int(max(1.0, dsl.safety))
    distances = _distance_transform(layers.obstacle)
    safety_cost = np.clip((safety_radius - distances) / safety_radius, 0.0, 1.0)
    grid += safety_cost.astype(np.float32) * 5.0

    semantic_costs: Dict[str, float] = {
        "water": 8.0,
        "human": 10.0,
        "glass": 6.0,
        "tree": 4.0,
        "vegetation": 3.5,
        "road": -2.0,
        "open_space": -1.5,
    }

    for label in dsl.avoid:
        for semantic_id in mapping.semantic_id_from_label(label):
            grid[layers.semantic_id == semantic_id] += semantic_costs.get(label, 3.0)

    for label in dsl.prefer:
        for semantic_id in mapping.semantic_id_from_label(label):
            grid[layers.semantic_id == semantic_id] += semantic_costs.get(label, -1.0)

    grid += float(uncertainty_weight) * layers.uncertainty
    grid = np.maximum(grid, 0.0)

    return Costmap(
        grid=grid,
        meta=CostmapMeta(resolution=resolution, origin=origin, frame_id=frame_id),
    )
