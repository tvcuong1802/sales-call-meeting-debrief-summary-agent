"""
src/nodes/main_node.py — CMN-C1-042

MainNode (slot: main). New-gen FunctionNode that folds the extraction + validation
passes into a single slot (no GraphNode wrapper, Cat-1 pattern):
  1. StructuredExtractNode  — single LLM call (retry-wrapped) → summary/crm_fields/email
  2. CRMSchemaValidateNode  — validate crm_fields against the CRM JSON Schema

Sub-nodes are DI'd and called via execute() (another template fold pattern). The LLM client
is injected via the Graph constructor; the sub-nodes load their own config from files.
Each sub-node self-guards on state["error"]; the slot also short-circuits when an error
is already set (the post_process S-3 gate still runs).
"""

from __future__ import annotations

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.invocation_context import TrustLevel

from src.utils.audit import emit_trace_event

from src.nodes.structured_extract import StructuredExtractNode
from src.nodes.crm_schema_validate import CRMSchemaValidateNode


class MainNode(FunctionNode):
    """Main slot — LLM structured extraction then CRM schema validation."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(
        self,
        llm_client: Any = None,
        structured_extract_node: StructuredExtractNode | None = None,
        crm_schema_validate_node: CRMSchemaValidateNode | None = None,
    ) -> None:
        super().__init__()
        self._structured_extract = structured_extract_node or StructuredExtractNode(llm_client=llm_client)
        self._crm_schema_validate = crm_schema_validate_node or CRMSchemaValidateNode()

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        # Short-circuit on an upstream fatal error — nothing to extract/validate.
        # pre_process already established there is no transcript. Running the extraction
        # fold would call the model on nothing and get filler back.
        if state.get("cannot_summarize"):
            emit_trace_event("debrief_extraction_skipped", {"reason": "no_transcript"}, state)
            return {
                "summary": "",
                "crm_fields": "{}",
                "cannot_summarize": True,
                "output_language": "",
                "status": __import__(
                    "framework.schemas.agent_status", fromlist=["AgentStatus"]
                ).AgentStatus.SUCCESS.value,
            }
        if state.get("error"):
            # Short-circuit: nothing extracted; keep an explicit empty summary.
            # S-4: this return path must be auditable too, not only the happy path.
            emit_trace_event("debrief_extraction_skipped", {"reason": "upstream_error"}, state)
            return {"error": state["error"], "summary": "", "crm_fields": "{}"}
        work = {**state, **self._structured_extract.execute(state)}
        work = {**work, **self._crm_schema_validate.execute(work)}
        # Partial-update dict of only main-stage fields (wheel contract — never the
        # full state). Set SUCCESS so route() reaches post_process (the S-3 sink);
        # any non-SUCCESS status routes straight to finalize.
        from framework.schemas.agent_status import AgentStatus

        update: dict[str, Any] = {
            "summary": work.get("summary", ""),
            "crm_fields": work.get("crm_fields", "{}"),
            "cannot_summarize": work.get("cannot_summarize", False),
            # The language the extraction resolved for THIS request. This dict is a
            # closed key-set -- anything the fold produced and this list omits is dropped
            # here, silently, at the composition boundary. That is what happened:
            # structured_extract returned "ja", the renderer received "en", and a
            # Japanese debrief kept English headings with no error anywhere. Measured by
            # spying on both ends, after two wrong guesses at the model and at the
            # detector (2026-09-03).
            "output_language": work.get("output_language", ""),
        }
        if work.get("error"):
            update["error"] = work["error"]
        else:
            update["status"] = AgentStatus.SUCCESS.value
        # S-4: the framework contract requires every execute() to emit at least one DOMAIN event.
        # node_start/node_complete/node_error come from BaseNode.__call__() and must not
        # be duplicated — this records what the extraction+validation fold produced.
        emit_trace_event(
            "debrief_extracted",
            {
                "summary_produced": bool(update.get("summary")),
                "cannot_summarize": bool(update.get("cannot_summarize")),
                "crm_validated": bool(work.get("crm_validated")),
                "failed": bool(update.get("error")),
            },
            state,
        )
        return update
