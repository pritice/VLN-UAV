"""Tests for synthetic asset generation."""
from pathlib import Path

import matplotlib.image as mpimg

from nav_core.utils.assets import generate_test_assets


def test_cases_generate_assets() -> None:
    assets_dir = Path(__file__).resolve().parent / "assets"
    assets = generate_test_assets(assets_dir, size=(64, 64))

    expected = {"t1_water.png", "t2_road.png", "t3_wall.png", "t7_nopath.png"}
    assert expected.issubset(set(assets.keys()))

    for name in expected:
        path = assets[name]
        assert path.exists()
        img = mpimg.imread(str(path))
        assert img.shape[0] == 64
        assert img.shape[1] == 64
