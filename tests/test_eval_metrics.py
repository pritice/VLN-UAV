"""Tests for evaluation metrics."""
import numpy as np

from scripts.eval_cases import compute_collision_rate, compute_path_overlap


def test_eval_overlap_and_collision() -> None:
    obstacle = np.zeros((5, 5), dtype=np.uint8)
    obstacle[2, 2] = 1

    path_a = [(0, 0), (1, 0), (2, 0)]
    path_b = [(1, 0), (2, 0), (3, 0)]

    overlap = compute_path_overlap(path_a, path_b)
    assert overlap == 2 / 4

    collision_path = [(2, 2), (3, 2)]
    collision_rate = compute_collision_rate(collision_path, obstacle)
    assert collision_rate == 0.5
