import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_correlation_id(config: dict[str, Any] | None) -> str:
    """Extract correlation_id from the framework config envelope.

    Tries ``config["configurable"]["invocation_context"].correlation_id``
    (production path), then ``config["configurable"]["session_id"]`` as a
    fallback (test / CI path). Returns "" when unavailable.

    Promoted from the former ``AgentBaseNode`` shim (removed in CR-0930): template
    nodes now inherit the framework ``FunctionNode`` directly and call this as a
    plain helper. The shim's ``_emit_trace_event`` stub was also dropped — the
    framework ``FunctionNode`` provides the real S-4 audit sink in production.
    """
    if config is None:
        return ""
    configurable = config.get("configurable", {})
    # Production path: InvocationContext object
    invocation_context = configurable.get("invocation_context")
    if invocation_context is not None:
        correlation_id = getattr(invocation_context, "correlation_id", None)
        if correlation_id:
            return str(correlation_id)
    # Fallback: session_id (test / CI path)
    session_id: str = configurable.get("session_id", "")
    return session_id
