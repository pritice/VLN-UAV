"""Tests for costmap generation."""
import numpy as np

from nav_core.contracts import GridLayers, InstructionDSL
from nav_core.costmap import build_costmap


def test_costmap_obstacle_penalty() -> None:
    obstacle = np.zeros((5, 5), dtype=np.uint8)
    obstacle[2, 2] = 1
    semantic_id = np.zeros((5, 5), dtype=np.int32)
    uncertainty = np.zeros((5, 5), dtype=np.float32)
    layers = GridLayers(semantic_id=semantic_id, obstacle=obstacle, uncertainty=uncertainty)

    dsl = InstructionDSL(goal="unknown")
    costmap = build_costmap(layers, dsl)

    assert costmap.grid[2, 2] > costmap.grid[0, 0]
