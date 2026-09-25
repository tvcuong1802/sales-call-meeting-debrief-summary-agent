"""PB-1b: the template's OWN audit wrapper must reach the platform audit logger.

Why this exists as a separate proof from PB-1.

PB-1 (`test_pb_invoke_order.py`) monkeypatches `framework.nodes.base_node.emit_trace_event` and
asserts the ORDER of gate/execute/gate calls plus the framework's own `node_start` /
`node_complete`. That is its job and it does it correctly -- but it never touches
`src/utils/audit.py`, the wrapper this template's nodes actually call. So a wrapper that invokes
the platform logger with the wrong arity, swallows the resulting `TypeError`, and writes to stderr
instead still passes PB-1. Ten repos in this fleet shipped exactly that: an audit trail present in
the source and absent at runtime.

This test closes that gap by asserting against the REAL sink -- the `agentcore.audit` logger --
with nothing patched, and by requiring a DOMAIN event. A lifecycle event would not do: those come
from the framework and arrive even when the template's wrapper is broken.
"""

from __future__ import annotations

import logging

from src.utils.audit import emit_trace_event


def test_template_audit_wrapper_reaches_the_platform_logger():
    """A domain event emitted through the template's wrapper must land on `agentcore.audit`.

    Deliberately asserts on the logger and not on stderr: the stderr path is the FALLBACK, and it
    is what a broken wrapper silently degrades to. Passing because the fallback fired would be the
    exact false green this test exists to prevent.
    """
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture()
    logger = logging.getLogger("agentcore.audit")
    prior_level, prior_propagate = logger.level, logger.propagate
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        # A minimal but VALID state: the platform logger reads trace_id / correlation_id /
        # session_id off it, so an empty dict would not exercise the real path.
        state = {
            "correlation_id": "pb1b-audit-sink",
            "session_id": "pb1b-session",
            "thread_id": "pb1b-thread",
            "trace_id": "pb1b-trace",
        }
        emit_trace_event("pb1b_probe_event", {"probe": True}, state)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior_level)
        logger.propagate = prior_propagate

    assert records, (
        "the template's audit wrapper emitted nothing to the `agentcore.audit` logger. "
        "A wrapper that calls the platform logger with the wrong arity raises TypeError, has it "
        "swallowed, and falls back to stderr -- which leaves this assertion empty while every "
        "other test still passes."
    )
    joined = " ".join(r.getMessage() for r in records)
    assert "pb1b_probe_event" in joined, f"the audit logger received records, but not this event: {joined[:200]!r}"
