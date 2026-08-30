"""Personal-workstation safety defaults for local long-running agents."""

from frontier_agent.core.runtime.loop._runaway import _RUNAWAY_MAX_RETRIES
from frontier_agent.core.runtime.loop.budget_consistency import check_context_budget


def test_default_allows_only_one_reasoning_runaway_retry() -> None:
    assert _RUNAWAY_MAX_RETRIES == 1


def test_64k_default_preserves_16k_output_with_a_4k_watchdog() -> None:
    problems = check_context_budget(
        max_len=65_536,
        max_input_tokens=49_152,
        max_tokens=16_384,
        reasoning_only_max_tokens=4_096,
        label="workstation-64k-default",
    )
    assert problems == []


def test_64k_explicit_32k_output_remains_safe_with_a_4k_watchdog() -> None:
    problems = check_context_budget(
        max_len=65_536,
        max_input_tokens=32_768,
        max_tokens=32_768,
        reasoning_only_max_tokens=4_096,
        label="workstation-64k-large-output",
    )
    assert problems == []
