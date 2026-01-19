"""Command line demo for language-conditioned navigation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nav_core.pipeline import run_pipeline
from nav_core.utils.io import build_observation
from nav_core.utils.vis import save_costmap, save_path_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to RGB image")
    parser.add_argument("--instruction", required=True, help="Natural language instruction")
    parser.add_argument("--outdir", required=True, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    observation = build_observation(args.image)
    dsl, costmap, plan = run_pipeline(observation, args.instruction)

    save_costmap(costmap.grid, str(outdir / "costmap.png"))
    save_path_overlay(observation.rgb, plan.path, str(outdir / "path_overlay.png"))

    payload = {
        "dsl": dsl.__dict__,
        "plan": {"cost": plan.cost, "stats": plan.stats, "path_length": len(plan.path)},
    }

    with (outdir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
