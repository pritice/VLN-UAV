"""Run a suite of language-conditioned navigation cases."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from nav_core.pipeline import run_pipeline
from nav_core.utils.assets import generate_test_assets
from nav_core.utils.io import build_observation
from nav_core.utils.vis import save_costmap, save_path_overlay


CASES: List[Dict[str, object]] = [
    {
        "id": "T0",
        "image": "t1_water.png",
        "instructionA": "avoid water and go to the top center",
        "instructionB": "go to the top center",
        "params": {"uncertainty_weight": 2.0},
    },
    {
        "id": "T1",
        "image": "t2_road.png",
        "instructionA": "prefer road and go to the top center",
        "instructionB": "avoid road and go to the top center",
        "params": {"uncertainty_weight": 1.0},
    },
    {
        "id": "T2",
        "image": "t3_wall.png",
        "instructionA": "go to the top center",
        "instructionB": "go to the left",
        "params": {"uncertainty_weight": 2.5},
    },
    {
        "id": "T3",
        "image": "t1_water.png",
        "instructionA": "avoid water and go to the left",
        "instructionB": "avoid water and go to the right",
        "params": {"uncertainty_weight": 1.5},
    },
    {
        "id": "T4",
        "image": "t2_road.png",
        "instructionA": "prefer road and go to the top center",
        "instructionB": None,
        "params": {"uncertainty_weight": 2.0},
    },
    {
        "id": "T5",
        "image": "t3_wall.png",
        "instructionA": "avoid tree and go to the right",
        "instructionB": "avoid tree and go to the left",
        "params": {"uncertainty_weight": 2.0},
    },
    {
        "id": "T6",
        "image": "t1_water.png",
        "instructionA": "go to the top center",
        "instructionB": None,
        "params": {"uncertainty_weight": 0.5},
    },
    {
        "id": "T7",
        "image": "t7_nopath.png",
        "instructionA": "go to the top center",
        "instructionB": "avoid water and go to the top center",
        "params": {"uncertainty_weight": 1.0},
    },
]


def _run_variant(
    case_id: str,
    variant: str,
    image_path: Path,
    instruction: str,
    params: Dict[str, float],
    outdir: Path,
) -> None:
    observation = build_observation(str(image_path))
    dsl, costmap, plan = run_pipeline(
        observation,
        instruction,
        costmap_params={"uncertainty_weight": params.get("uncertainty_weight", 2.0)},
    )

    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "costmap.npy", costmap.grid)
    save_costmap(costmap.grid, str(outdir / "costmap.png"))
    save_path_overlay(observation.rgb, plan.path, str(outdir / "path.png"))

    with (outdir / "dsl.json").open("w", encoding="utf-8") as f:
        json.dump(dsl.__dict__, f, ensure_ascii=False, indent=2)

    with (outdir / "plan.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "path": plan.path,
                "cost": plan.cost,
                "stats": plan.stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with (outdir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "case_id": case_id,
                "variant": variant,
                "image": str(image_path),
                "instruction": instruction,
                "params": params,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assets_dir = repo_root / "tests" / "assets"
    assets = generate_test_assets(assets_dir)

    outputs_root = repo_root / "outputs" / "cases"

    for case in CASES:
        case_id = str(case["id"])
        image_name = str(case["image"])
        image_path = assets.get(image_name, assets_dir / image_name)
        params = case.get("params", {})
        instruction_a = str(case["instructionA"])
        instruction_b = case.get("instructionB")

        _run_variant(
            case_id,
            "A",
            image_path,
            instruction_a,
            params if isinstance(params, dict) else {},
            outputs_root / case_id / "A",
        )

        if isinstance(instruction_b, str):
            _run_variant(
                case_id,
                "B",
                image_path,
                instruction_b,
                params if isinstance(params, dict) else {},
                outputs_root / case_id / "B",
            )


if __name__ == "__main__":
    main()
