"""Tests for A* planner."""
import numpy as np

from nav_core.contracts import Costmap, CostmapMeta
from nav_core.planner import AStarPlanner


def test_astar_finds_path_around_obstacle() -> None:
    grid = np.ones((10, 10), dtype=np.float32)
    grid[5, :] = 1e6
    grid[5, 5] = 1.0
    costmap = Costmap(grid=grid, meta=CostmapMeta(0.1, (0.0, 0.0), "map"))
    planner = AStarPlanner(connectivity=4)
    start = (5, 9)
    goal = (5, 0)
    plan = planner.plan(costmap, start, goal)

    assert plan.path
    assert plan.path[0] == start
    assert plan.path[-1] == goal
