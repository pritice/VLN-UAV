"""Data contracts for the navigation pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Observation:
    """Sensor observation input for navigation."""

    rgb: np.ndarray
    timestamp: Optional[float] = None
    intrinsics: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None


@dataclass
class InstructionDSL:
    """Structured instruction used by the navigation stack."""

    goal: str
    avoid: List[str] = field(default_factory=list)
    prefer: List[str] = field(default_factory=list)
    safety: float = 3.0
    fallback: str = "top_center"


@dataclass
class GridLayers:
    """Grid layers derived from perception."""

    semantic_id: np.ndarray
    obstacle: np.ndarray
    uncertainty: np.ndarray


@dataclass
class CostmapMeta:
    """Metadata for costmap grids."""

    resolution: float
    origin: Tuple[float, float]
    frame_id: str


@dataclass
class Costmap:
    """Costmap container."""

    grid: np.ndarray
    meta: CostmapMeta


@dataclass
class Plan:
    """Path planning output."""

    path: List[Tuple[int, int]]
    cost: float
    stats: Dict[str, float]
