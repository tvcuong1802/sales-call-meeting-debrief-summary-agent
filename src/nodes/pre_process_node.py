"""
src/nodes/pre_process_node.py — CMN-C1-042

PreProcessNode (slot: pre_process). New-gen FunctionNode that folds the two
input-stage passes into a single slot (no GraphNode wrapper, Cat-1 pattern):
  1. TranscriptCleanNode — S-2 PII masking + transcript normalization (entry node;
                           sets invocation_id / audit_input_hash)
  2. NuanceClassifyNode  — Japanese business-nuance annotation

Sub-nodes are DI'd and called via execute() (another template fold pattern). Each sub-node
self-guards on state["error"] (returns unchanged), so a fatal S-2 error in
TranscriptCleanNode propagates without NuanceClassify reprocessing.
Config is loaded by the sub-nodes from config files (no node_config injection needed).
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.invocation_context import TrustLevel

from src.utils.audit import emit_trace_event

from src.nodes.transcript_clean import TranscriptCleanNode
from src.nodes.nuance_classify import NuanceClassifyNode


class PreProcessNode(FunctionNode):
    """Pre-process slot — transcript clean (S-2) then Japanese nuance annotation."""

    # S-1 trust gate (ADR-006): enforced by BaseNode.__call__() before execute().
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(
        self,
        transcript_clean_node: TranscriptCleanNode | None = None,
        nuance_classify_node: NuanceClassifyNode | None = None,
    ) -> None:
        super().__init__()
        self._transcript_clean = transcript_clean_node or TranscriptCleanNode()
        self._nuance_classify = nuance_classify_node or NuanceClassifyNode()

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        # Does the message carry a transcript at all? Judged HERE, on the text as
        # received, because that is the only point where it is still the user's. By the
        # time it reaches the extraction node the nuance pass has stamped labels onto it,
        # so a message of pure punctuation arrives carrying letters that were never
        # typed -- and three attempts at the check further downstream could not tell the
        # two apart. A predicate that needs a third fix is in the wrong place, not
        # missing a fourth patch.
        # Both fields: the framework seeds `user_input` on invoke, while direct callers
        # and the state factory use `raw_transcript`. Reading only one made a normal
        # transcript look empty and short-circuited it.
        message = str(state.get("user_input") or state.get("raw_transcript") or "")
        # Only a message that HAS content but no letters. A blank one is already rejected
        # downstream as S1_INPUT_REJECTED, and that error record is a contract callers
        # depend on -- this branch must not take it over.
        if message.strip() and not re.search(r"[A-Za-z\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf]", message):
            emit_trace_event("pre_process_no_transcript", {"reason": "no_letters_in_message"}, state)
            return {
                "cleaned_transcript": "",
                "annotated_transcript": "",
                "cannot_summarize": True,
                "output_language": "",
            }

        work = {**state, **self._transcript_clean.execute(state)}
        work = {**work, **self._nuance_classify.execute(work)}
        # Partial-update dict only (wheel contract — node_history is accumulate-reducer).
        update = {
            k: work[k]
            for k in (
                "cleaned_transcript",
                "annotated_transcript",
                "pii_masked",
                "pii_mask_count",
                "invocation_id",
                "audit_input_hash",
            )
            if k in work
        }
        if work.get("error"):
            update["error"] = work["error"]
        # S-4: the framework contract requires every execute() to emit at least one DOMAIN event,
        # regardless of side effects. node_start/node_complete/node_error are emitted by
        # BaseNode.__call__() and must not be duplicated here — this records what the fold
        # actually produced: whether the transcript was masked and annotated.
        emit_trace_event(
            "input_stage_prepared",
            {
                "pii_masked": bool(update.get("pii_masked")),
                "pii_mask_count": update.get("pii_mask_count", 0),
                "nuance_annotated": "annotated_transcript" in update,
                "halted_on_error": bool(update.get("error")),
            },
            state,
        )
        return update
