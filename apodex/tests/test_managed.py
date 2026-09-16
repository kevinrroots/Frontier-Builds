import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from apodex.managed import (
    ManagedApprover,
    ManagedBudgetObserver,
    ManagedState,
    _digest,
    _load_managed_resume_prompt,
    load_managed_request,
    mark_managed_cli_failure,
)
from frontier_agent.core.loop_types import TurnContext


def _request(root: Path, **overrides):
    operation_id = overrides.pop("operation_id", "op-123")
    body = {
        "schema_version": 1,
        "protocol": "frontier-managed-v1",
        "action": "submit",
        "operation_id": operation_id,
        "task_id": "task-123",
        "attempt_id": "attempt-123",
        "correlation_id": "correlation-123",
        "session_id": "managed-op-123",
        "task_summary": "Inspect the project",
        "acceptance_criteria": "Return verified findings",
        "workspace": str(root / "workspace"),
        "time_limit_seconds": 60,
        "token_limit": 5000,
        "approval_policy": "owner_required_each_risky_tool",
        "binding_id": "frontieragent-operational-default",
        "binding_digest": "a" * 64,
        "runtime_profile": "frontieragent-operational-default",
        "launch_index": 1,
        "issued_at": "2026-09-08T00:00:00+00:00",
        **overrides,
    }
    body["request_digest"] = _digest(body)
    directory = root / operation_id
    directory.mkdir(parents=True)
    (root / "workspace").mkdir(exist_ok=True)
    path = directory / "request-0001.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return body, path


def test_request_is_digest_bound_and_confined(tmp_path, monkeypatch):
    payload, path = _request(tmp_path)
    monkeypatch.setenv("APODEX_MANAGED_ROOT", str(tmp_path))
    request, loaded_path = load_managed_request(str(path))
    assert loaded_path == path
    assert request.request_digest == payload["request_digest"]

    payload["task_summary"] = "changed after issue"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_managed_request(str(path))


def test_managed_approval_requires_bound_agentos_decision(tmp_path, monkeypatch):
    _, path = _request(tmp_path)
    monkeypatch.setenv("APODEX_MANAGED_ROOT", str(tmp_path))
    request, request_path = load_managed_request(str(path))
    state = ManagedState(request, request_path)
    approver = ManagedApprover(state, time.monotonic() + 5)
    async def exercise():
        pending = asyncio.create_task(
            approver.confirm(
                "write_file",
                "result.txt",
                "writes a file",
                preview="new contents",
                preview_kind="diff",
            )
        )
        approval_path = path.parent / "approval-0001-0001.request.json"
        for _ in range(20):
            if approval_path.exists():
                break
            await asyncio.sleep(0.01)
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
        decision = {
            "schema_version": 1,
            "protocol": "frontier-managed-v1",
            "operation_id": request.operation_id,
            "request_digest": request.request_digest,
            "approval_name": approval["approval_name"],
            "approval_digest": approval["approval_digest"],
            "decision": "approved",
            "decided_at": "2026-09-08T00:01:00+00:00",
            "native_approval_id": "agno-approval-123",
        }
        decision["decision_digest"] = _digest(decision)
        (path.parent / "approval-0001-0001.decision.json").write_text(
            json.dumps(decision), encoding="utf-8"
        )
        return await pending

    result = asyncio.run(exercise())
    assert result.approved is True
    assert result.remember is False
    assert approver.auto_approve is False


def test_managed_budget_stops_before_another_turn(tmp_path, monkeypatch):
    _, path = _request(tmp_path)
    monkeypatch.setenv("APODEX_MANAGED_ROOT", str(tmp_path))
    request, request_path = load_managed_request(str(path))
    guard = ManagedBudgetObserver(ManagedState(request, request_path), token_limit=100)
    context = TurnContext(
        turn=1,
        max_turns=2,
        task_id="task-123",
        role_id="react_agent",
        ai_text="",
        thinking="",
        tool_calls=[],
        messages=[],
        usage={"prompt_tokens": 80, "completion_tokens": 20},
        metadata={},
    )
    intervention = asyncio.run(guard.on_llm_response(context))
    assert intervention is not None
    assert intervention.stop_reason == "managed_token_limit"


def test_managed_cli_preflight_failure_is_terminal_and_monotonic(
    tmp_path, monkeypatch
):
    _, path = _request(tmp_path)
    monkeypatch.setenv("APODEX_MANAGED_ROOT", str(tmp_path))
    request, request_path = load_managed_request(str(path))
    state = ManagedState(request, request_path)
    state.write("starting")
    mark_managed_cli_failure(str(path), "FRONTIER_MANAGED_CLI_EXIT_2")
    status = json.loads((path.parent / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["sequence"] == 2
    mark_managed_cli_failure(str(path), "SHOULD_NOT_REPLACE_TERMINAL")
    assert json.loads(
        (path.parent / "status.json").read_text(encoding="utf-8")
    ) == status


def test_managed_cli_exit_preserves_resumable_interrupted_state(
    tmp_path, monkeypatch
):
    _, path = _request(tmp_path)
    monkeypatch.setenv("APODEX_MANAGED_ROOT", str(tmp_path))
    request, request_path = load_managed_request(str(path))
    state = ManagedState(request, request_path)
    state.write("interrupted")
    status = json.loads((path.parent / "status.json").read_text(encoding="utf-8"))
    mark_managed_cli_failure(str(path), "FRONTIER_MANAGED_CLI_EXIT_130")
    assert json.loads(
        (path.parent / "status.json").read_text(encoding="utf-8")
    ) == status


def test_managed_resume_reloads_digest_bound_original_prompt(
    tmp_path, monkeypatch
):
    original_payload, original_path = _request(
        tmp_path, session_id="managed-resume-session"
    )
    _, resume_path = _request(
        tmp_path,
        operation_id="op-resume",
        action="resume",
        session_id="managed-resume-session",
        launch_index=2,
    )
    monkeypatch.setenv("APODEX_MANAGED_ROOT", str(tmp_path))
    resume_request, _ = load_managed_request(str(resume_path))
    session = SimpleNamespace(
        _managed_resume_request_path=str(original_path)
    )
    assert (
        _load_managed_resume_prompt(session, resume_request)
        == f"{original_payload['task_summary']}\n\n"
        f"Acceptance criteria:\n{original_payload['acceptance_criteria']}"
    )


def test_managed_resume_rejects_missing_or_mismatched_source(
    tmp_path, monkeypatch
):
    _, original_path = _request(
        tmp_path, session_id="different-session"
    )
    _, resume_path = _request(
        tmp_path,
        operation_id="op-resume",
        action="resume",
        session_id="managed-resume-session",
        launch_index=2,
    )
    monkeypatch.setenv("APODEX_MANAGED_ROOT", str(tmp_path))
    resume_request, _ = load_managed_request(str(resume_path))
    with pytest.raises(FileNotFoundError):
        _load_managed_resume_prompt(SimpleNamespace(), resume_request)
    with pytest.raises(ValueError, match="binding mismatch"):
        _load_managed_resume_prompt(
            SimpleNamespace(
                _managed_resume_request_path=str(original_path)
            ),
            resume_request,
        )
