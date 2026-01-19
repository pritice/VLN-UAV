import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PathResult:
    path: List[Tuple[int, int]]
    cost: float


def astar(occupancy: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], extra_cost: Optional[np.ndarray] = None) -> Optional[PathResult]:
    h, w = occupancy.shape
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return None
    if occupancy[sy, sx] == 1 or occupancy[gy, gx] == 1:
        return None

    def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    frontier: List[Tuple[float, Tuple[int, int]]] = []
    heapq.heappush(frontier, (0.0, (sx, sy)))
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {(sx, sy): None}
    cost_so_far: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == (gx, gy):
            break
        cx, cy = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if occupancy[ny, nx] == 1:
                continue
            step_cost = 1.0
            if extra_cost is not None:
                step_cost += float(extra_cost[ny, nx])
            new_cost = cost_so_far[current] + step_cost
            if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                cost_so_far[(nx, ny)] = new_cost
                priority = new_cost + heuristic((nx, ny), (gx, gy))
                heapq.heappush(frontier, (priority, (nx, ny)))
                came_from[(nx, ny)] = current

    if (gx, gy) not in came_from:
        return None

    path = []
    node: Optional[Tuple[int, int]] = (gx, gy)
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return PathResult(path=path, cost=cost_so_far[(gx, gy)])
