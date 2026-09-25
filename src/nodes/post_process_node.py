"""
src/nodes/post_process_node.py — CMN-C1-042

PostProcessNode (slot: post_process). New-gen FunctionNode. NON-SUPPRESSIBLE.
Always executes (incl. error paths). Wraps the terminal S-3 gate:
  OutputFormatNode — content-safety / credential-leak scan + final output formatting;
                     runs even when state["error"] is set (design §6.3, §8.3).

Sub-node is DI'd and called via execute() (another template fold pattern). This node is the
sole graph sink — no path bypasses the S-3 output gate.
"""

from __future__ import annotations

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.invocation_context import TrustLevel

from src.utils.audit import emit_trace_event

from src.nodes.output_format import OutputFormatNode


class PostProcessNode(FunctionNode):
    """Post-process slot — non-suppressible S-3 output gate + final formatting."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, output_format_node: OutputFormatNode | None = None) -> None:
        super().__init__()
        self._output_format = output_format_node or OutputFormatNode()

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        # S-3 gate runs unconditionally — even on the error path (non-suppressible).
        merged = {**state, **self._output_format.execute(state)}
        # Partial-update dict only (wheel contract — never the full state; node_history
        # is an accumulate-reducer field). Surface `result` as invoke() output and mark
        # SUCCESS so route() reaches this sink (any non-SUCCESS status → finalize).
        from framework.schemas.agent_status import AgentStatus

        update: dict[str, Any] = {
            "output_json": merged.get("output_json", ""),
            "audit_output_hash": merged.get("audit_output_hash", ""),
            # get_output() surfaces state["result"] as invoke() `output`. Keep it a
            # primitive (JSON str) so State stays msgpack-safe (no dict in State).
            # The chat surface reads `result`. Markdown when it exists, the JSON only as
            # the fallback for a run that never reached the renderer.
            "result": merged.get("output_markdown") or merged.get("output_json", ""),
        }
        if not merged.get("error"):
            update["status"] = AgentStatus.SUCCESS.value
        else:
            update["error"] = merged["error"]
        # S-4: the framework contract requires every execute() to emit at least one DOMAIN event.
        # node_start/node_complete/node_error come from BaseNode.__call__() and must not
        # be duplicated — this records that the non-suppressible S-3 sink ran and what it
        # produced, including on the error path (this node never suppresses).
        emit_trace_event(
            "output_gate_applied",
            {
                "output_emitted": bool(update.get("output_json")),
                "output_hashed": bool(update.get("audit_output_hash")),
                "on_error_path": bool(merged.get("error")),
            },
            state,
        )
        return update
