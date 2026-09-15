"""Native file protocol for AgentOS-governed FrontierAgent runs.

This is deliberately not a terminal adapter.  AgentOS writes one immutable
request, FrontierAgent writes atomic status and approval records, and the
ordinary FrontierAgent session remains the owner of checkpoints and artifacts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import signal
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from apodex.observers import Approver, Decision
from apodex.render import Renderer
from frontier_agent.core.loop_types import BaseObserver, Intervention, TurnContext

PROTOCOL = "frontier-managed-v1"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_MAX_APPROVAL_PREVIEW = 16_000


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    temporary.replace(path)


class ManagedRequest(BaseModel):
    schema_version: Literal[1]
    protocol: Literal["frontier-managed-v1"]
    action: Literal["submit", "resume"]
    operation_id: str = Field(min_length=1, max_length=128)
    task_id: str = Field(min_length=1, max_length=128)
    attempt_id: str = Field(min_length=1, max_length=128)
    correlation_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    task_summary: str = Field(min_length=1, max_length=4000)
    acceptance_criteria: str = Field(min_length=1, max_length=4000)
    workspace: str = Field(min_length=1, max_length=4000)
    time_limit_seconds: int = Field(ge=1, le=86_400)
    token_limit: int = Field(ge=1, le=1_000_000)
    approval_policy: Literal["owner_required_each_risky_tool"]
    binding_id: str = Field(min_length=1, max_length=200)
    binding_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_profile: str = Field(min_length=1, max_length=200)
    launch_index: int = Field(ge=1, le=1000)
    issued_at: str = Field(min_length=1, max_length=80)
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity_and_digest(self) -> ManagedRequest:
        for name in ("operation_id", "task_id", "attempt_id", "correlation_id"):
            if _ID_RE.fullmatch(str(getattr(self, name))) is None:
                raise ValueError(f"{name} is not a safe managed identity")
        if _ID_RE.fullmatch(self.session_id) is None:
            raise ValueError("session_id is not a safe managed identity")
        if not Path(self.workspace).is_absolute() or Path(self.workspace) == Path("/"):
            raise ValueError("workspace must be an absolute non-root path")
        payload = self.model_dump(mode="json", exclude={"request_digest"})
        if _digest(payload) != self.request_digest:
            raise ValueError("managed request digest mismatch")
        return self

    @property
    def prompt(self) -> str:
        return (
            f"{self.task_summary.strip()}\n\n"
            "Acceptance criteria:\n"
            f"{self.acceptance_criteria.strip()}"
        )


def load_managed_request(path_value: str) -> tuple[ManagedRequest, Path]:
    configured = os.environ.get("APODEX_MANAGED_ROOT", "").strip()
    if not configured:
        raise ValueError("APODEX_MANAGED_ROOT is not configured")
    root = Path(configured).expanduser().resolve(strict=True)
    raw_path = Path(path_value).expanduser()
    if raw_path.is_symlink():
        raise ValueError("managed request must not be a symlink")
    path = raw_path.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("managed request escapes the configured root") from exc
    if not path.is_file():
        raise ValueError("managed request must be a regular non-symlink file")
    if path.stat().st_size > 32_768:
        raise ValueError("managed request exceeds the fixed size limit")
    raw = json.loads(path.read_text(encoding="utf-8"))
    request = ManagedRequest.model_validate(raw)
    if path.parent.name != request.operation_id:
        raise ValueError("managed request directory does not match operation_id")
    expected_workspace = os.environ.get("APODEX_HOST_PROJECT_DIR", "").strip()
    if (
        expected_workspace
        and Path(expected_workspace).resolve() != Path(request.workspace).resolve()
    ):
        raise ValueError("managed request workspace does not match the mounted project")
    return request, path


class ManagedState:
    def __init__(self, request: ManagedRequest, request_path: Path) -> None:
        self.request = request
        self.directory = request_path.parent
        self.sequence = 0
        status_path = self.directory / "status.json"
        try:
            if status_path.is_symlink() or status_path.stat().st_size > 32_768:
                raise ValueError("invalid managed status file")
            existing = json.loads(status_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                raise ValueError("invalid managed status record")
            body = dict(existing)
            status_digest = body.pop("status_digest", None)
            if (
                status_digest == _digest(body)
                and body.get("request_digest") == request.request_digest
                and body.get("launch_index") == request.launch_index
            ):
                self.sequence = max(0, int(body.get("sequence", 0)))
        except (OSError, TypeError, ValueError):
            pass

    def write(self, state: str, **details: Any) -> dict[str, Any]:
        self.sequence += 1
        record = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "operation_id": self.request.operation_id,
            "task_id": self.request.task_id,
            "correlation_id": self.request.correlation_id,
            "session_id": self.request.session_id,
            "launch_index": self.request.launch_index,
            "request_digest": self.request.request_digest,
            "state": state,
            "sequence": self.sequence,
            "updated_at": _now(),
            **details,
        }
        body = dict(record)
        record["status_digest"] = _digest(body)
        _atomic_json(self.directory / "status.json", record)
        return record

    def cancelled(self) -> bool:
        path = self.directory / "cancel.json"
        try:
            if path.is_symlink() or path.stat().st_size > 4096:
                return False
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return False
        if not isinstance(payload, dict):
            return False
        cancel_digest = payload.pop("cancel_digest", None)
        if not isinstance(cancel_digest, str) or _digest(payload) != cancel_digest:
            return False
        return bool(
            payload.get("operation_id") == self.request.operation_id
            and payload.get("request_digest") == self.request.request_digest
            and payload.get("cancel") is True
        )


def mark_managed_cli_failure(path_value: str, error_code: str) -> None:
    """Best-effort terminal record for failures before the managed runner starts."""
    try:
        request, request_path = load_managed_request(path_value)
        state = ManagedState(request, request_path)
        status_path = state.directory / "status.json"
        if status_path.exists():
            if status_path.is_symlink() or status_path.stat().st_size > 32_768:
                raise ValueError("invalid managed status file")
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if not isinstance(status, dict):
                raise ValueError("invalid managed status record")
            body = dict(status)
            status_digest = body.pop("status_digest", None)
            if (
                status_digest == _digest(body)
                and body.get("request_digest") == request.request_digest
                and body.get("launch_index") == request.launch_index
                and body.get("state")
                in {
                    "approval_denied",
                    "cancelled",
                    "completed",
                    "failed",
                    "interrupted",
                    "timed_out",
                }
            ):
                return
        state.write("failed", error_code=error_code)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return


class ManagedApprover(Approver):
    """Approval gate whose decisions arrive through AgentOS-owned files."""

    def __init__(self, state: ManagedState, deadline: float) -> None:
        super().__init__(auto_approve=False, auto_for_me=False, interactive=False)
        self.state = state
        self.deadline = deadline
        self.count = 0
        self.last_outcome = ""

    async def confirm(
        self,
        name: str,
        target: str,
        reason: str,
        *,
        dangerous: str = "",
        preview: str = "",
        preview_kind: str = "",
    ) -> Decision:
        self.count += 1
        approval_name = (
            f"approval-{self.state.request.launch_index:04d}-{self.count:04d}"
        )
        request_path = self.state.directory / f"{approval_name}.request.json"
        decision_path = self.state.directory / f"{approval_name}.decision.json"
        body: dict[str, Any] = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "operation_id": self.state.request.operation_id,
            "request_digest": self.state.request.request_digest,
            "approval_name": approval_name,
            "tool_name": str(name)[:200],
            "target": str(target)[:1000],
            "reason": str(reason)[:2000],
            "dangerous": str(dangerous)[:2000],
            "preview_kind": str(preview_kind)[:80],
            "preview": str(preview)[:_MAX_APPROVAL_PREVIEW],
            "created_at": _now(),
        }
        approval_digest = _digest(body)
        record = {**body, "approval_digest": approval_digest}
        if request_path.exists():
            if request_path.is_symlink() or request_path.stat().st_size > 32_768:
                self.last_outcome = "approval_identity_conflict"
                return Decision(False)
            existing = json.loads(request_path.read_text(encoding="utf-8"))
            if existing != record:
                self.last_outcome = "approval_identity_conflict"
                return Decision(False)
        else:
            _atomic_json(request_path, record)
        self.state.write(
            "waiting_approval",
            approval_name=approval_name,
            approval_digest=approval_digest,
            approval_request=request_path.name,
        )
        while time.monotonic() < self.deadline:
            if self.state.cancelled():
                self.last_outcome = "cancelled"
                return Decision(False)
            try:
                if decision_path.is_symlink() or decision_path.stat().st_size > 4096:
                    raise ValueError("invalid decision file")
                decision = json.loads(decision_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                await asyncio.sleep(0.25)
                continue
            except (OSError, TypeError, ValueError):
                self.last_outcome = "invalid_approval_decision"
                return Decision(False)
            expected = {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "operation_id": self.state.request.operation_id,
                "request_digest": self.state.request.request_digest,
                "approval_name": approval_name,
                "approval_digest": approval_digest,
                "decision": decision.get("decision"),
                "decided_at": decision.get("decided_at"),
                "native_approval_id": decision.get("native_approval_id"),
            }
            if decision != {**expected, "decision_digest": _digest(expected)}:
                self.last_outcome = "invalid_approval_decision"
                return Decision(False)
            if decision["decision"] not in {"approved", "denied"}:
                self.last_outcome = "invalid_approval_decision"
                return Decision(False)
            self.last_outcome = str(decision["decision"])
            self.state.write("running", approval_name=approval_name)
            return Decision(decision["decision"] == "approved")
        self.last_outcome = "approval_timeout"
        return Decision(False)


class ManagedBudgetObserver(BaseObserver):
    critical = True

    def __init__(self, state: ManagedState, token_limit: int) -> None:
        self.state = state
        self.token_limit = token_limit
        self.tokens = 0

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None:
        usage = ctx.usage or {}
        self.tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.tokens += int(usage.get("completion_tokens", 0) or 0)
        if self.tokens >= self.token_limit:
            return Intervention(stop_reason="managed_token_limit")
        return None

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        if self.state.cancelled():
            return Intervention(stop_reason="managed_cancelled")
        return None


class ManagedRenderer(Renderer):
    def __init__(self) -> None:
        super().__init__(theme="mono", color=False)
        self.final_text = ""
        self.failure_text = ""
        self.incomplete_reason = ""

    def final(
        self,
        text: str,
        *,
        turns: int = 0,
        tool_calls: int = 0,
        stopped_by: str = "",
    ) -> None:
        self.final_text = text
        super().final(text, turns=turns, tool_calls=tool_calls, stopped_by=stopped_by)

    def error(self, msg: str) -> None:
        self.failure_text = msg
        super().error(msg)

    def llm_failure(self, msg: str, *, configuration_error: bool = False) -> None:
        self.failure_text = msg
        super().llm_failure(msg, configuration_error=configuration_error)

    def incomplete(
        self,
        text: str,
        *,
        turns: int = 0,
        tool_calls: int = 0,
        stopped_by: str = "",
    ) -> None:
        self.final_text = text
        self.incomplete_reason = stopped_by or "incomplete"
        super().incomplete(
            text,
            turns=turns,
            tool_calls=tool_calls,
            stopped_by=stopped_by,
        )


def _write_final(state: ManagedState, text: str) -> tuple[str, int]:
    encoded = text.encode()
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "operation_id": state.request.operation_id,
        "session_id": state.request.session_id,
        "text": text,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "created_at": _now(),
    }
    _atomic_json(state.directory / "final.json", result)
    return str(result["sha256"]), int(result["bytes"])


async def run_managed_request(
    path_value: str,
    *,
    cfg: Any,
    max_turns: int,
    mode: str,
) -> int:
    """Run one validated request with no TTY and no auto-approval path."""
    try:
        request, request_path = load_managed_request(path_value)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: invalid managed request: {exc}", file=os.sys.stderr)
        return 2
    state = ManagedState(request, request_path)
    state.write("starting", worker_pid=os.getpid())
    deadline = time.monotonic() + request.time_limit_seconds

    from apodex.session import TerminalSession
    from apodex.session_state import load_session_state

    renderer = ManagedRenderer()
    cfg.max_tokens = min(int(cfg.max_tokens), request.token_limit)
    session = TerminalSession(
        cfg=cfg,
        cwd=os.getcwd(),
        renderer=renderer,
        auto_approve=False,
        max_turns=max_turns,
        interactive=False,
        mode=mode,
        session_id=request.session_id,
        plan_mode=False,
    )
    approver = ManagedApprover(state, deadline)
    session.approver = approver
    session.plan_state.active = False
    session.rules = None
    budget = ManagedBudgetObserver(state, request.token_limit)
    session.managed_observers = [budget]
    if request.action == "resume":
        restored = load_session_state(request.session_id)
        if restored is None:
            state.write("failed", error_code="FRONTIER_RESUME_STATE_MISSING")
            return 1
        session.restore(restored)
        budget.tokens = session.usage.total
        if budget.tokens >= request.token_limit:
            state.write(
                "failed",
                error_code="FRONTIER_MANAGED_TOKEN_BUDGET_EXHAUSTED",
                token_usage=session.usage.total,
            )
            return 1

    loop = asyncio.get_running_loop()
    current = asyncio.current_task()
    installed_signal_handler = False
    if current is not None:
        try:
            loop.add_signal_handler(signal.SIGTERM, current.cancel)
            installed_signal_handler = True
        except (NotImplementedError, RuntimeError):
            pass
    state.write("running", worker_pid=os.getpid())
    try:
        remaining = max(0.1, deadline - time.monotonic())
        prompt = request.prompt if request.action == "submit" else ""
        await asyncio.wait_for(session.run_task(prompt), timeout=remaining)
    except TimeoutError:
        session._persist()
        state.write("timed_out", token_usage=session.usage.total)
        return 124
    except asyncio.CancelledError:
        session._persist()
        terminal = "cancelled" if state.cancelled() else "interrupted"
        state.write(terminal, token_usage=session.usage.total)
        return 130
    finally:
        if installed_signal_handler:
            loop.remove_signal_handler(signal.SIGTERM)

    if state.cancelled() or approver.last_outcome == "cancelled":
        terminal = "cancelled"
    elif approver.last_outcome in {
        "approval_timeout",
        "approval_identity_conflict",
        "invalid_approval_decision",
    }:
        terminal = "failed"
    elif approver.last_outcome == "denied":
        terminal = "approval_denied"
    elif renderer.failure_text:
        terminal = "failed"
    elif renderer.incomplete_reason:
        terminal = "interrupted"
    elif renderer.final_text:
        terminal = "completed"
    else:
        terminal = "interrupted"
    run_dir = os.environ.get(
        "APODEX_HOST_RUN_DIR", os.environ.get("APODEX_RUN_DIR", "")
    )
    details: dict[str, Any] = {
        "token_usage": session.usage.total,
        "run_dir": run_dir,
        "trace_path": str(Path(run_dir) / "trace.jsonl") if run_dir else "",
        "outputs_dir": str(Path(run_dir) / "outputs") if run_dir else "",
    }
    if renderer.final_text:
        final_digest, final_bytes = _write_final(state, renderer.final_text)
        details.update(final_digest=final_digest, final_bytes=final_bytes)
    if renderer.failure_text:
        details["error_sha256"] = hashlib.sha256(renderer.failure_text.encode()).hexdigest()
    if approver.last_outcome:
        details["approval_outcome"] = approver.last_outcome
    state.write(terminal, **details)
    return 0 if terminal == "completed" else 1


__all__ = [
    "PROTOCOL",
    "ManagedApprover",
    "ManagedBudgetObserver",
    "ManagedRequest",
    "ManagedState",
    "load_managed_request",
    "mark_managed_cli_failure",
    "run_managed_request",
]
