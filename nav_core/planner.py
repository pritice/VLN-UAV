"""Grid-based path planners."""
from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np

from nav_core.contracts import Costmap, InstructionDSL, Plan


class AStarPlanner:
    """A* planner operating on a costmap grid."""

    def __init__(self, connectivity: int = 8) -> None:
        if connectivity not in (4, 8):
            raise ValueError("Connectivity must be 4 or 8")
        self.connectivity = connectivity

    def plan(
        self, costmap: Costmap, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> Plan:
        """Compute a path with A* search."""

        grid = costmap.grid
        h, w = grid.shape

        def in_bounds(node: Tuple[int, int]) -> bool:
            x, y = node
            return 0 <= x < w and 0 <= y < h

        def neighbors(node: Tuple[int, int]) -> List[Tuple[int, int]]:
            x, y = node
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if self.connectivity == 8:
                steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            return [(x + dx, y + dy) for dx, dy in steps if in_bounds((x + dx, y + dy))]

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

        open_set: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, start))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score = {start: 0.0}
        visited = 0

        obstacle_threshold = 1e5

        while open_set:
            _, current = heapq.heappop(open_set)
            visited += 1

            if current == goal:
                break

            for nxt in neighbors(current):
                if grid[nxt[1], nxt[0]] >= obstacle_threshold:
                    continue
                step_cost = float(grid[nxt[1], nxt[0]])
                tentative = g_score[current] + step_cost
                if tentative < g_score.get(nxt, float("inf")):
                    came_from[nxt] = current
                    g_score[nxt] = tentative
                    f_score = tentative + heuristic(nxt, goal)
                    heapq.heappush(open_set, (f_score, nxt))

        path = _reconstruct_path(came_from, start, goal)
        total_cost = g_score.get(goal, float("inf"))
        stats = {"expanded": float(visited), "path_length": float(len(path))}
        return Plan(path=path, cost=total_cost, stats=stats)


def _reconstruct_path(
    came_from: Dict[Tuple[int, int], Tuple[int, int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> List[Tuple[int, int]]:
    if goal not in came_from and goal != start:
        return []

    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def default_start_goal(costmap: Costmap, dsl: InstructionDSL) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Infer start/goal positions from the costmap and DSL."""

    h, w = costmap.grid.shape
    start = (w // 2, h - 1)
    goal = (w // 2, 0)

    if any(token in dsl.goal.lower() for token in ["left", "左"]):
        goal = (w // 4, 0)
    elif any(token in dsl.goal.lower() for token in ["right", "右"]):
        goal = (3 * w // 4, 0)

    if any(token in dsl.goal.lower() for token in ["bottom", "下"]):
        goal = (goal[0], h - 1)
    elif any(token in dsl.goal.lower() for token in ["center", "中"]):
        goal = (w // 2, h // 2)

    return start, goal
