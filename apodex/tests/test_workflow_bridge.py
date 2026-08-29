from pathlib import Path

from apodex.docker import model_runtime_env
from apodex.task_runner import _native_workflow_profile_overrides
from workflows.stateful_react_agent.nodes.main_agent import (
    _direct_worktree_root,
)


def test_terminal_coding_root_overrides_run_private_workspace(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    scratch = tmp_path / "scratch"
    project.mkdir()
    scratch.mkdir()

    selected = _direct_worktree_root(
        {
            "metadata": {
                "coding_workspace_root": str(project),
            },
        },
        "task-1",
        str(scratch),
    )

    assert selected == project


def test_nonterminal_workflow_keeps_run_private_workspace(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    selected = _direct_worktree_root(
        {"metadata": {}},
        "task-2",
        str(scratch),
    )

    assert selected == scratch


def test_cli_turn_budget_becomes_native_workflow_override() -> None:
    assert _native_workflow_profile_overrides(80) == {
        "agent": {
            "main_max_turns": 80,
        },
    }


def test_docker_forwards_native_workflow_input_budget() -> None:
    assert model_runtime_env(
        {
            "OPENAI_CONTEXT_WINDOW": "65536",
            "OPENAI_MAX_INPUT_TOKENS": "49152",
            "OPENAI_MAX_TOKENS": "16384",
        },
    ) == [
        "-e",
        "OPENAI_CONTEXT_WINDOW",
        "-e",
        "OPENAI_MAX_INPUT_TOKENS",
        "-e",
        "OPENAI_MAX_TOKENS",
    ]
