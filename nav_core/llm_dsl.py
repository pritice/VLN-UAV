"""Deterministic parser that maps natural language to a DSL."""
from __future__ import annotations

import re
from typing import List

from nav_core.contracts import InstructionDSL

_AVOID_KEYWORDS = {
    "water": "water",
    "人": "human",
    "human": "human",
    "玻璃": "glass",
    "glass": "glass",
    "树": "tree",
    "tree": "tree",
}

_PREFER_KEYWORDS = {
    "road": "road",
    "道路": "road",
    "open": "open_space",
    "空地": "open_space",
}

_GOAL_PATTERNS = [
    r"go to\s*([\w\u4e00-\u9fa5]+)",
    r"前往\s*([\w\u4e00-\u9fa5]+)",
    r"到\s*([\w\u4e00-\u9fa5]+)",
    r"靠近\s*([\w\u4e00-\u9fa5]+)",
]


def parse_instruction(text: str) -> InstructionDSL:
    """Parse a natural language instruction into a DSL structure.

    Args:
        text: Natural language instruction.

    Returns:
        InstructionDSL with defaults populated.
    """

    lowered = text.lower()
    avoid: List[str] = []
    prefer: List[str] = []

    for key, value in _AVOID_KEYWORDS.items():
        if key in lowered or key in text:
            avoid.append(value)

    for key, value in _PREFER_KEYWORDS.items():
        if key in lowered or key in text:
            prefer.append(value)

    goal = "unknown"
    for pattern in _GOAL_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            goal = match.group(1)
            break

    return InstructionDSL(
        goal=goal,
        avoid=sorted(set(avoid)),
        prefer=sorted(set(prefer)),
        safety=3.0,
        fallback="top_center",
    )
