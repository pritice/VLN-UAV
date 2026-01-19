"""Main pipeline orchestration."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from nav_core.contracts import Costmap, InstructionDSL, Observation, Plan
from nav_core.costmap import build_costmap
from nav_core.llm_dsl import parse_instruction
from nav_core.mapping import generate_layers
from nav_core.planner import AStarPlanner, default_start_goal


def run_pipeline(
    observation: Observation,
    instruction: str,
    *,
    costmap_params: Optional[Dict[str, float]] = None,
) -> Tuple[InstructionDSL, Costmap, Plan]:
    """Run the minimal navigation pipeline.

    Args:
        observation: Input observation with RGB image.
        instruction: Natural language instruction.

    Returns:
        Parsed DSL, costmap, and plan.
    """

    dsl = parse_instruction(instruction)
    layers = generate_layers(observation.rgb)
    costmap_kwargs = costmap_params or {}
    costmap = build_costmap(layers, dsl, **costmap_kwargs)
    planner = AStarPlanner(connectivity=8)
    start, goal = default_start_goal(costmap, dsl)
    plan = planner.plan(costmap, start, goal)
    return dsl, costmap, plan
