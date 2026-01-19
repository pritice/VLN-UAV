"""Tests for contracts and DSL parsing."""
from nav_core.llm_dsl import parse_instruction


def test_dsl_fields_complete() -> None:
    dsl = parse_instruction("avoid water and go to exit")
    assert isinstance(dsl.goal, str)
    assert isinstance(dsl.avoid, list)
    assert isinstance(dsl.prefer, list)
    assert isinstance(dsl.safety, float)
    assert isinstance(dsl.fallback, str)
    assert "water" in dsl.avoid
