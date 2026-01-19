import argparse
import json
import pathlib
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from vlnuav.nav import astar


def parse_point(text: str) -> Tuple[int, int]:
    vals = [int(v) for v in text.split(",")]
    if len(vals) != 2:
        raise ValueError("Point must be x,y")
    return vals[0], vals[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="A* navigation demo on BEV map")
    parser.add_argument("--map-dir", default="outputs")
    parser.add_argument("--start", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--output", default="outputs/nav_path.png")
    args = parser.parse_args()

    map_dir = pathlib.Path(args.map_dir)
    occupancy = np.load(map_dir / "bev_occupancy.npy")
    height = np.load(map_dir / "bev_height_max.npy")
    start = parse_point(args.start)
    goal = parse_point(args.goal)
    height_cost = np.nan_to_num(height, nan=0.0, posinf=0.0, neginf=0.0)
    height_cost = (height_cost - height_cost.min())
    if height_cost.max() > 0:
        height_cost = height_cost / height_cost.max()

    result = astar(occupancy, start, goal, extra_cost=height_cost)
    if result is None:
        raise RuntimeError("No path found")

    plt.figure(figsize=(6, 6))
    plt.imshow(occupancy, cmap="gray")
    xs = [p[0] for p in result.path]
    ys = [p[1] for p in result.path]
    plt.plot(xs, ys, "r-")
    plt.scatter([start[0], goal[0]], [start[1], goal[1]], c=["green", "blue"], s=30)
    plt.title(f"Path cost: {result.cost:.2f}")
    plt.axis("off")
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    meta_path = map_dir / "meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        meta["nav_demo"] = {"start": start, "goal": goal, "cost": result.cost}
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)


if __name__ == "__main__":
    main()
