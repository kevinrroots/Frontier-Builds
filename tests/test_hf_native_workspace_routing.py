"""Regression coverage for direct/native HF session workspace selection."""

from pathlib import Path

from workflows.stateful_react_agent.nodes.main_agent import (
    _direct_worktree_root,
)


def test_explicit_native_workspace_wins_over_private_trial_state() -> None:
    state = {
        "metadata": {
            "_trial_dir": "/session/state",
            "coding_workspace_root": "/session/workspace",
        },
    }

    assert _direct_worktree_root(state, "task", "/mounted/workspace") == Path(
        "/session/workspace",
    )


def test_native_workspace_uses_mount_default_without_explicit_root() -> None:
    assert _direct_worktree_root(
        {"metadata": {"_trial_dir": "/session/state"}},
        "task",
        "/mounted/workspace",
    ) == Path("/mounted/workspace")
