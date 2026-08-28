"""Tests for CLI-to-workflow turn-budget publication."""

from __future__ import annotations

import pytest

from apodex.cli import publish_turn_override


def test_publish_turn_override_replaces_profile_environment() -> None:
    environ = {"MAIN_MAX_TURNS": "600"}

    publish_turn_override(6, environ=environ)

    assert environ["MAIN_MAX_TURNS"] == "6"


def test_publish_turn_override_creates_profile_environment() -> None:
    environ: dict[str, str] = {}

    publish_turn_override(1, environ=environ)

    assert environ == {"MAIN_MAX_TURNS": "1"}


@pytest.mark.parametrize("value", [0, -1, -100])
def test_publish_turn_override_rejects_invalid_budget(value: int) -> None:
    environ = {"MAIN_MAX_TURNS": "600"}

    with pytest.raises(ValueError, match="at least 1"):
        publish_turn_override(value, environ=environ)

    assert environ == {"MAIN_MAX_TURNS": "600"}
