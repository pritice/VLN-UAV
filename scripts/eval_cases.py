"""Evaluate navigation case outputs."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from nav_core import mapping
from nav_core.costmap import _distance_transform


@dataclass
class VariantMetrics:
    """Per-variant metrics computed for a single run."""

    collision_rate: Optional[float]
    min_dist_to_obstacle: Optional[float]
    uncertainty_on_path: Optional[float]


@dataclass
class PairMetrics:
    """Metrics comparing variant A and B."""

    costmap_diff_mean: Optional[float]
    path_overlap: Optional[float]
    path_length_delta: Optional[float]
    semantic_cost_delta: Dict[str, Optional[float]]


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_path(path: Path) -> List[Tuple[int, int]]:
    data = _load_json(path)
    return [tuple(point) for point in data.get("path", [])]


def _path_set(path_points: List[Tuple[int, int]]) -> set[Tuple[int, int]]:
    return set(path_points)


def compute_path_overlap(path_a: List[Tuple[int, int]], path_b: List[Tuple[int, int]]) -> Optional[float]:
    """Compute Jaccard overlap between two paths."""

    if not path_a or not path_b:
        return None
    set_a = _path_set(path_a)
    set_b = _path_set(path_b)
    union = set_a | set_b
    if not union:
        return None
    return len(set_a & set_b) / len(union)


def compute_collision_rate(path: List[Tuple[int, int]], obstacle: np.ndarray) -> Optional[float]:
    """Collision rate is the fraction of path cells that intersect obstacles."""

    if not path:
        return None
    collisions = 0
    for x, y in path:
        if 0 <= y < obstacle.shape[0] and 0 <= x < obstacle.shape[1]:
            if obstacle[y, x] > 0:
                collisions += 1
    return collisions / len(path)


def compute_min_dist_to_obstacle(
    path: List[Tuple[int, int]], obstacle: np.ndarray
) -> Optional[float]:
    """Minimum distance from path to any obstacle."""

    if not path:
        return None
    dist = _distance_transform(obstacle)
    values = [dist[y, x] for x, y in path if 0 <= y < dist.shape[0] and 0 <= x < dist.shape[1]]
    if not values:
        return None
    return float(min(values))


def compute_uncertainty_on_path(
    path: List[Tuple[int, int]], uncertainty: np.ndarray
) -> Optional[float]:
    """Average uncertainty along the path."""

    if not path:
        return None
    values = [
        float(uncertainty[y, x])
        for x, y in path
        if 0 <= y < uncertainty.shape[0] and 0 <= x < uncertainty.shape[1]
    ]
    if not values:
        return None
    return float(np.mean(values))


def compute_semantic_cost_delta(
    cost_a: np.ndarray, cost_b: np.ndarray, semantic_id: np.ndarray
) -> Dict[str, Optional[float]]:
    """Compute mean cost differences over semantic regions.

    Returns a dict with keys for water/road/vegetation. If a region mask is empty,
    the value is None.
    """

    results: Dict[str, Optional[float]] = {}
    for label in ["water", "road", "vegetation"]:
        ids = mapping.semantic_id_from_label(label)
        if not ids:
            results[label] = None
            continue
        mask = np.isin(semantic_id, list(ids))
        if not mask.any():
            results[label] = None
            continue
        delta = cost_a[mask] - cost_b[mask]
        results[label] = float(np.mean(delta))
    return results


def compute_variant_metrics(
    path: List[Tuple[int, int]], obstacle: np.ndarray, uncertainty: np.ndarray
) -> VariantMetrics:
    return VariantMetrics(
        collision_rate=compute_collision_rate(path, obstacle),
        min_dist_to_obstacle=compute_min_dist_to_obstacle(path, obstacle),
        uncertainty_on_path=compute_uncertainty_on_path(path, uncertainty),
    )


def compute_pair_metrics(
    cost_a: np.ndarray,
    cost_b: np.ndarray,
    path_a: List[Tuple[int, int]],
    path_b: List[Tuple[int, int]],
    semantic_id: np.ndarray,
) -> PairMetrics:
    return PairMetrics(
        costmap_diff_mean=float(np.mean(np.abs(cost_a - cost_b))) if cost_a.shape == cost_b.shape else None,
        path_overlap=compute_path_overlap(path_a, path_b),
        path_length_delta=float(len(path_a) - len(path_b)) if path_a or path_b else None,
        semantic_cost_delta=compute_semantic_cost_delta(cost_a, cost_b, semantic_id),
    )


def _load_variant(case_dir: Path, variant: str) -> Optional[Dict[str, object]]:
    variant_dir = case_dir / variant
    if not variant_dir.exists():
        return None
    costmap_path = variant_dir / "costmap.npy"
    plan_path = variant_dir / "plan.json"
    config_path = variant_dir / "run_config.json"
    if not costmap_path.exists() or not plan_path.exists() or not config_path.exists():
        return None

    costmap = np.load(costmap_path)
    path = _load_path(plan_path)
    config = _load_json(config_path)

    image_path = Path(str(config.get("image", "")))
    if image_path.exists():
        from nav_core.utils.io import load_rgb

        rgb = load_rgb(str(image_path))
        layers = mapping.generate_layers(rgb)
    else:
        layers = None

    return {
        "costmap": costmap,
        "path": path,
        "config": config,
        "layers": layers,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    outputs_root = repo_root / "outputs" / "cases"

    metrics: Dict[str, Dict[str, object]] = {}

    for case_dir in sorted(outputs_root.glob("T*")):
        case_id = case_dir.name
        variant_a = _load_variant(case_dir, "A")
        variant_b = _load_variant(case_dir, "B")

        case_metrics: Dict[str, object] = {}

        if variant_a and variant_a["layers"] is not None:
            layers = variant_a["layers"]
            variant_metrics = compute_variant_metrics(variant_a["path"], layers.obstacle, layers.uncertainty)
            case_metrics["variant_A"] = asdict(variant_metrics)
        else:
            case_metrics["variant_A"] = asdict(VariantMetrics(None, None, None))

        if variant_b and variant_b["layers"] is not None:
            layers = variant_b["layers"]
            variant_metrics = compute_variant_metrics(variant_b["path"], layers.obstacle, layers.uncertainty)
            case_metrics["variant_B"] = asdict(variant_metrics)
        else:
            case_metrics["variant_B"] = asdict(VariantMetrics(None, None, None))

        if variant_a and variant_b and variant_a["layers"] is not None:
            pair_metrics = compute_pair_metrics(
                variant_a["costmap"],
                variant_b["costmap"],
                variant_a["path"],
                variant_b["path"],
                variant_a["layers"].semantic_id,
            )
            case_metrics["pair"] = {
                "costmap_diff_mean": pair_metrics.costmap_diff_mean,
                "path_overlap": pair_metrics.path_overlap,
                "path_length_delta": pair_metrics.path_length_delta,
                "semantic_cost_delta": pair_metrics.semantic_cost_delta,
            }
        else:
            case_metrics["pair"] = {
                "costmap_diff_mean": None,
                "path_overlap": None,
                "path_length_delta": None,
                "semantic_cost_delta": {
                    "water": None,
                    "road": None,
                    "vegetation": None,
                },
            }

        metrics[case_id] = case_metrics

    metrics_payload = {
        "definitions": {
            "costmap_diff_mean": "mean(|C_A - C_B|) over all cells; None if variant missing",
            "semantic_cost_delta": "mean(C_A - C_B) over semantic masks for water/road/vegetation",
            "path_overlap": "Jaccard overlap of path cell sets; None if a path is empty",
            "path_length_delta": "len(path_A) - len(path_B); None if both missing",
            "collision_rate": "fraction of path cells intersecting obstacles; None if no path",
            "min_dist_to_obstacle": "minimum distance from path to any obstacle; None if no path",
            "uncertainty_on_path": "mean uncertainty values along path; None if no path",
        },
        "metrics": metrics,
    }

    outputs_root.mkdir(parents=True, exist_ok=True)
    metrics_json = outputs_root / "metrics.json"
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    metrics_csv = outputs_root / "metrics.csv"
    fieldnames = [
        "case_id",
        "costmap_diff_mean",
        "path_overlap",
        "path_length_delta",
        "collision_rate_A",
        "collision_rate_B",
        "min_dist_to_obstacle_A",
        "min_dist_to_obstacle_B",
        "uncertainty_on_path_A",
        "uncertainty_on_path_B",
        "semantic_cost_delta_water",
        "semantic_cost_delta_road",
        "semantic_cost_delta_vegetation",
    ]

    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case_id, case_metrics in metrics.items():
            pair = case_metrics.get("pair", {})
            variant_a = case_metrics.get("variant_A", {})
            variant_b = case_metrics.get("variant_B", {})
            semantic_delta = pair.get("semantic_cost_delta", {}) if isinstance(pair, dict) else {}
            writer.writerow(
                {
                    "case_id": case_id,
                    "costmap_diff_mean": pair.get("costmap_diff_mean"),
                    "path_overlap": pair.get("path_overlap"),
                    "path_length_delta": pair.get("path_length_delta"),
                    "collision_rate_A": variant_a.get("collision_rate"),
                    "collision_rate_B": variant_b.get("collision_rate"),
                    "min_dist_to_obstacle_A": variant_a.get("min_dist_to_obstacle"),
                    "min_dist_to_obstacle_B": variant_b.get("min_dist_to_obstacle"),
                    "uncertainty_on_path_A": variant_a.get("uncertainty_on_path"),
                    "uncertainty_on_path_B": variant_b.get("uncertainty_on_path"),
                    "semantic_cost_delta_water": semantic_delta.get("water"),
                    "semantic_cost_delta_road": semantic_delta.get("road"),
                    "semantic_cost_delta_vegetation": semantic_delta.get("vegetation"),
                }
            )


if __name__ == "__main__":
    main()
