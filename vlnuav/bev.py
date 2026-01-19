from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class GridConfig:
    size_x: float
    size_y: float
    resolution: float
    min_height: float
    max_height: float


@dataclass
class BEVMap:
    occupancy: np.ndarray
    height_max: np.ndarray
    semantic: Optional[np.ndarray]
    origin: Tuple[float, float]
    resolution: float


def _grid_shape(config: GridConfig) -> Tuple[int, int]:
    nx = int(np.ceil(config.size_x / config.resolution))
    ny = int(np.ceil(config.size_y / config.resolution))
    return nx, ny


def init_bev(config: GridConfig) -> BEVMap:
    nx, ny = _grid_shape(config)
    occupancy = np.zeros((ny, nx), dtype=np.uint8)
    height_max = np.full((ny, nx), fill_value=-np.inf, dtype=np.float32)
    return BEVMap(occupancy=occupancy, height_max=height_max, semantic=None, origin=(-config.size_x / 2, -config.size_y / 2), resolution=config.resolution)


def update_bev(
    bev: BEVMap,
    points_w: np.ndarray,
    heights: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> None:
    if points_w.size == 0:
        return
    xs = points_w[:, 0]
    ys = points_w[:, 1]
    origin_x, origin_y = bev.origin
    ix = np.floor((xs - origin_x) / bev.resolution).astype(int)
    iy = np.floor((ys - origin_y) / bev.resolution).astype(int)
    valid = (ix >= 0) & (iy >= 0) & (iy < bev.occupancy.shape[0]) & (ix < bev.occupancy.shape[1])
    if not np.any(valid):
        return
    ix = ix[valid]
    iy = iy[valid]
    h = heights[valid]
    bev.occupancy[iy, ix] = 1
    for x_idx, y_idx, z in zip(ix, iy, h):
        if z > bev.height_max[y_idx, x_idx]:
            bev.height_max[y_idx, x_idx] = z
    if labels is not None:
        if bev.semantic is None:
            bev.semantic = np.full(bev.occupancy.shape, fill_value=-1, dtype=np.int32)
        labels = labels[valid]
        for x_idx, y_idx, label in zip(ix, iy, labels):
            if label < 0:
                continue
            bev.semantic[y_idx, x_idx] = int(label)


def points_from_depth(depth: np.ndarray, intrinsics: Tuple[float, float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = intrinsics
    h, w = depth.shape
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    zs = depth
    valid = zs > 0
    xs = xs[valid]
    ys = ys[valid]
    zs = zs[valid]
    x = (xs - cx) * zs / fx
    y = (ys - cy) * zs / fy
    points = np.stack([x, y, zs], axis=1)
    return points, zs


def transform_points(points: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    homog = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    pts_w = (t_wc @ homog.T).T
    return pts_w[:, :3]


def merge_maps(maps: Iterable[BEVMap]) -> BEVMap:
    maps = list(maps)
    if not maps:
        raise ValueError("No maps to merge")
    merged = init_bev(GridConfig(
        size_x=maps[0].occupancy.shape[1] * maps[0].resolution,
        size_y=maps[0].occupancy.shape[0] * maps[0].resolution,
        resolution=maps[0].resolution,
        min_height=-10,
        max_height=10,
    ))
    merged.origin = maps[0].origin
    for bev in maps:
        merged.occupancy = np.maximum(merged.occupancy, bev.occupancy)
        merged.height_max = np.maximum(merged.height_max, bev.height_max)
        if bev.semantic is not None:
            if merged.semantic is None:
                merged.semantic = np.full(merged.occupancy.shape, fill_value=-1, dtype=np.int32)
            mask = bev.semantic >= 0
            merged.semantic[mask] = bev.semantic[mask]
    return merged


def semantic_masks(semantic: np.ndarray, things_ids: List[int], stuff_ids: List[int]) -> Dict[str, np.ndarray]:
    masks = {}
    if semantic is None:
        return masks
    masks["things"] = np.isin(semantic, things_ids).astype(np.uint8)
    masks["stuff"] = np.isin(semantic, stuff_ids).astype(np.uint8)
    return masks
