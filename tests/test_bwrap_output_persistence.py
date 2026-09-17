from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


def test_bwrap_tool_sandbox_receives_all_persistent_mounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plugins.tools._sandbox as sb

    workspace = tmp_path / "workspace"
    outputs = tmp_path / "outputs"
    inputs = tmp_path / "inputs"
    project = tmp_path / "project"
    for directory in (workspace, outputs, inputs, project):
        directory.mkdir()

    monkeypatch.setenv("SANDBOX_BACKEND", "bwrap")
    monkeypatch.setenv("FRONTIER_AGENT_WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(outputs))
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", str(inputs))
    monkeypatch.setenv("FRONTIER_AGENT_PROJECT_DIR", str(project))
    monkeypatch.setattr(sb, "_sandbox", None)
    monkeypatch.setattr(sb, "_sandbox_identity", None)
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_create(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(sb, "_create_provisioned_sandbox", fake_create)

    assert sb.get_sandbox() is sentinel
    assert captured["use_e2b"] is False
    assert captured["workspace"] == str(workspace)
    assert set(captured["binds"]) == {
        (str(outputs), "/outputs", False),
        (str(inputs), "/inputs", True),
        (str(project), str(project), False),
    }


def test_bwrap_tool_sandbox_rebinds_after_session_output_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plugins.tools._sandbox as sb

    workspace = tmp_path / "workspace"
    first_outputs = tmp_path / "first-outputs"
    second_outputs = tmp_path / "second-outputs"
    inputs = tmp_path / "inputs"
    for directory in (workspace, first_outputs, second_outputs, inputs):
        directory.mkdir()

    monkeypatch.setenv("SANDBOX_BACKEND", "bwrap")
    monkeypatch.setenv("FRONTIER_AGENT_WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(first_outputs))
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", str(inputs))
    monkeypatch.delenv("FRONTIER_AGENT_PROJECT_DIR", raising=False)
    monkeypatch.setattr(sb, "_sandbox", None)
    monkeypatch.setattr(sb, "_sandbox_identity", None)
    created: list[object] = []

    class Commands:
        @staticmethod
        def run(*_args: object, **_kwargs: object) -> object:
            return type("Result", (), {"exit_code": 0})()

    class FakeSandbox:
        commands = Commands()

        def __init__(self, binds: tuple[tuple[str, str, bool], ...]) -> None:
            self.binds = binds
            self.killed = False

        def kill(self) -> None:
            self.killed = True

    def fake_create(**kwargs: object) -> object:
        sandbox = FakeSandbox(kwargs["binds"])
        created.append(sandbox)
        return sandbox

    monkeypatch.setattr(sb, "_create_provisioned_sandbox", fake_create)

    first = sb.get_sandbox()
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(second_outputs))
    second = sb.get_sandbox()

    assert first is not second
    assert first.killed is True
    assert len(created) == 2
    assert (str(first_outputs), "/outputs", False) in first.binds
    assert (str(second_outputs), "/outputs", False) in second.binds


@pytest.mark.skipif(
    not __import__("plugins.tools._sandbox", fromlist=["bwrap_available"]).bwrap_available(),
    reason="bubblewrap is unavailable",
)
def test_create_file_persists_through_explicit_bwrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plugins.tools._sandbox as sb
    from plugins.tools.create_file import create_file

    workspace = tmp_path / "workspace"
    outputs = tmp_path / "outputs"
    inputs = tmp_path / "inputs"
    project = tmp_path / "project"
    for directory in (workspace, outputs, inputs, project):
        directory.mkdir()

    monkeypatch.setenv("SANDBOX_BACKEND", "bwrap")
    monkeypatch.setenv("FRONTIER_AGENT_WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(outputs))
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", str(inputs))
    monkeypatch.setenv("FRONTIER_AGENT_PROJECT_DIR", str(project))
    monkeypatch.setattr(sb, "_sandbox", None)
    monkeypatch.setattr(sb, "_sandbox_identity", None)

    try:
        result = asyncio.run(create_file.ainvoke({
            "path": "/outputs/approved.txt",
            "content": "BWRAP_OUTPUT_PERSISTENCE_SENTINEL",
        }))
    finally:
        sb.close_sandbox()

    approved = outputs / "approved.txt"
    assert "created txt" in result
    assert approved.read_text(encoding="utf-8") == "BWRAP_OUTPUT_PERSISTENCE_SENTINEL"


def test_active_bwrap_strategy_synchronizes_file_tool_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    import apodex.sandbox as sandbox
    from apodex.sandbox import BWRAP, Strategy, set_active_strategy

    monkeypatch.delenv("SANDBOX_BACKEND", raising=False)
    monkeypatch.setattr(sandbox, "_active", None)

    set_active_strategy(Strategy(BWRAP, "test"))

    assert os.environ["SANDBOX_BACKEND"] == "bwrap"


def test_cli_bash_bwrap_receives_session_persistent_mounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apodex.sandbox as shell_sb
    import plugins.tools._sandbox as tool_sb

    project = tmp_path / "project"
    outputs = tmp_path / "outputs"
    inputs = tmp_path / "inputs"
    for directory in (project, outputs, inputs):
        directory.mkdir()

    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(outputs))
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", str(inputs))
    monkeypatch.setattr(shell_sb, "_bwrap_sandbox", None)
    monkeypatch.setattr(shell_sb, "_bwrap_sandbox_identity", None)
    captured: dict[str, object] = {}

    class FakeSandbox:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def kill(self) -> None:
            raise AssertionError("fresh sandbox must not be killed")

    monkeypatch.setattr(tool_sb, "BwrapSandbox", FakeSandbox)

    sandbox = shell_sb._get_bwrap_sandbox(str(project))

    assert isinstance(sandbox, FakeSandbox)
    assert captured["workspace"] == str(project)
    assert set(captured["binds"]) == {
        (str(project), str(project), False),
        (str(outputs), "/outputs", False),
        (str(inputs), "/inputs", True),
    }


def test_cli_bash_bwrap_rebinds_after_stable_output_alias_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apodex.sandbox as shell_sb
    import plugins.tools._sandbox as tool_sb

    project = tmp_path / "project"
    first_outputs = tmp_path / "first-outputs"
    second_outputs = tmp_path / "second-outputs"
    output_alias = tmp_path / "outputs-link"
    for directory in (project, first_outputs, second_outputs):
        directory.mkdir()
    output_alias.symlink_to(first_outputs, target_is_directory=True)

    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(output_alias))
    monkeypatch.delenv("FRONTIER_AGENT_INPUTS_DIR", raising=False)
    monkeypatch.setattr(shell_sb, "_bwrap_sandbox", None)
    monkeypatch.setattr(shell_sb, "_bwrap_sandbox_identity", None)
    created: list[object] = []

    class FakeSandbox:
        def __init__(self, **kwargs: object) -> None:
            self.binds = kwargs["binds"]
            self.killed = False
            created.append(self)

        def kill(self) -> None:
            self.killed = True

    monkeypatch.setattr(tool_sb, "BwrapSandbox", FakeSandbox)

    first = shell_sb._get_bwrap_sandbox(str(project))
    output_alias.unlink()
    output_alias.symlink_to(second_outputs, target_is_directory=True)
    second = shell_sb._get_bwrap_sandbox(str(project))

    assert first is not second
    assert first.killed is True
    assert len(created) == 2
    assert (str(first_outputs), "/outputs", False) in first.binds
    assert (str(second_outputs), "/outputs", False) in second.binds


@pytest.mark.skipif(
    not __import__("plugins.tools._sandbox", fromlist=["bwrap_available"]).bwrap_available(),
    reason="bubblewrap is unavailable",
)
def test_bash_persists_through_explicit_cli_bwrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import apodex.sandbox as shell_sb
    from apodex.sandbox import BWRAP, Strategy, run_shell

    project = tmp_path / "project"
    outputs = tmp_path / "outputs"
    inputs = tmp_path / "inputs"
    for directory in (project, outputs, inputs):
        directory.mkdir()

    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(outputs))
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", str(inputs))
    monkeypatch.setattr(shell_sb, "_bwrap_sandbox", None)
    monkeypatch.setattr(shell_sb, "_bwrap_sandbox_identity", None)

    try:
        rc, stdout, stderr = asyncio.run(run_shell(
            "printf BASH_BWRAP_OUTPUT_PERSISTENCE_SENTINEL "
            "> /outputs/bash-approved.txt",
            str(project),
            30,
            Strategy(BWRAP, "test"),
        ))
    finally:
        sandbox = shell_sb._bwrap_sandbox
        if sandbox is not None:
            sandbox.kill()
        shell_sb._bwrap_sandbox = None
        shell_sb._bwrap_sandbox_identity = None

    approved = outputs / "bash-approved.txt"
    assert rc == 0, (stdout, stderr)
    assert approved.read_text(encoding="utf-8") == (
        "BASH_BWRAP_OUTPUT_PERSISTENCE_SENTINEL"
    )
