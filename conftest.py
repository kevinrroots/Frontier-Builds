import os
from collections.abc import Iterator

import pytest

# CP09_PROCESS_RUNTIME_STATE_ISOLATION_V1
_PROCESS_RUNTIME_ENV_KEYS = (
    "SANDBOX_BACKEND",
    "APODEX_SPILL_DIR",
    "FRONTIER_AGENT_WORKSPACE_DIR",
    "FRONTIER_AGENT_OUTPUTS_DIR",
    "FRONTIER_AGENT_INPUTS_DIR",
    "FRONTIER_AGENT_PROJECT_DIR",
)


@pytest.fixture(autouse=True)
def _restore_process_runtime_state_after_each_test() -> Iterator[None]:
    """Prevent in-process CLI tests from contaminating later test cases."""
    import apodex.sandbox as strategy_state
    import plugins.tools._sandbox as shared_state

    environment = {key: os.environ.get(key) for key in _PROCESS_RUNTIME_ENV_KEYS}
    active = strategy_state._active
    shared = shared_state._sandbox
    shared_identity = shared_state._sandbox_identity
    try:
        yield
    finally:
        for key, value in environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        strategy_state._active = active
        shared_state._sandbox = shared
        shared_state._sandbox_identity = shared_identity

