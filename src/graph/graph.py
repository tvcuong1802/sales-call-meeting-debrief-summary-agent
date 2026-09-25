"""
src/graph/graph.py — CMN-C1-042

New-gen graph composition for the Sales Call & Meeting Debrief Summary Agent.
Graph(AgentBaseGraph) fills the three mandatory slots in register_nodes() (no
GraphNode wrapper — Cat-1 pattern, design §0):
  pre_process  → PreProcessNode   (TranscriptClean S-2 + NuanceClassify)
  main         → MainNode         (StructuredExtract LLM + CRMSchemaValidate)
  post_process → PostProcessNode  (OutputFormat — S-3 terminal gate; sole sink)
The framework owns compile()/invoke()/routing; this class adds no custom execute/invoke.
The LLM client is injected here and forwarded to the main slot's StructuredExtractNode;
the sub-nodes load their own config (config/config.yaml, nuance / schema files).
"""

from __future__ import annotations

import re

from typing import Any

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.schemas.agent_status import AgentStatus

from src.nodes.pre_process_node import PreProcessNode
from src.nodes.main_node import MainNode
from src.nodes.post_process_node import PostProcessNode
from src.schemas.state import DebriefState


def _injected_client(config: dict[str, Any] | None) -> Any:
    """A caller-supplied LLM CLIENT from config, or None.

    `llm:` in this repo's config.yaml is a block of model settings (model, max_tokens,
    retry), not a client -- so a bare `config.get("llm")` handed the graph a dict, and
    the first call raised "'dict' object has no attribute 'invoke'". A client is
    something that can be CALLED; a settings mapping is not one.
    """
    value = (config or {}).get("llm_client")
    if value is None:
        candidate = (config or {}).get("llm")
        value = None if isinstance(candidate, dict) else candidate
    return value


class Graph(AgentBaseGraph):
    """Level-1 Cat-1 agent — CMN-C1-042 (Sales Call & Meeting Debrief)."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._llm_client = _injected_client(config)
        super().__init__(config)

    @property
    def name(self) -> str:
        return "cmn_c1_042"

    @property
    def state_schema(self) -> type:
        # Persist domain working-fields across nodes (wheel contract).
        return DebriefState

    def register_nodes(self) -> None:
        super().register_nodes()  # injects initialize / finalize (real SDK)
        self._nodes = getattr(self, "_nodes", None) or {}
        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = MainNode(llm_client=self._llm_client)
        self._nodes["post_process"] = PostProcessNode()

    def get_output(self, state: Any) -> dict[str, Any]:
        """The framework envelope, with a liability trailer on the payload.

        The envelope shape is kept: the Marketplace runner and the Stage-5 evidence
        script both read it as a dict (`status`, `output`). Only the payload changes.
        """
        envelope = _with_disclaimer(dict(super().get_output(state)), state)
        # The nodes set `status` only when there is no `error`, so a degraded run -- the
        # model unreachable, everything else computed -- left InitializeNode's `pending`
        # in place. `normalize_terminal_output()` raises on any status but SUCCESS, so the
        # reader got "agent failed" while the deterministic extraction sat in the payload.
        # `error` stays exactly where it is: it is the record of what went wrong. `status`
        # is the transport, and it must say "there is something to read" when there is.
        if str(envelope.get("status")) != AgentStatus.SUCCESS.value and str(envelope.get("output") or "").strip():
            envelope["status"] = AgentStatus.SUCCESS.value
        return envelope


def _decided_language(state: Any) -> str | None:
    """The language this reader wants, or None when the message has none to read.

    `answer_language` is the decision the intake call makes from the message as
    received. Falling back to the script of the message covers the paths where S-2
    rejected the input in the gate, so `execute()` never ran: the agent still knows WHO
    it is talking to even when it will not act on WHAT they said.

    Returning None means bilingual, and that is now reserved for its one honest case --
    punctuation, digits, an empty line. Passing None unconditionally, as this file used
    to, gave every reader whose request failed an English block with a Japanese
    translation stapled underneath, which reads as an agent that never worked out which
    of the two it was for.
    """
    if not isinstance(state, dict):
        return None
    code = str(state.get("answer_language") or "").strip().lower()[:2]
    if code in ("en", "ja"):
        return code
    for key in ("user_input", "raw_query", "validated_input", "normalized_query"):
        value = state.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        japanese = sum(1 for ch in value if "\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff")
        # Latin WORDS, not characters: a Japanese message quoting an API key or a long id
        # used to read as English -- one 26-character token outweighed fourteen kana.
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z']*", value) if 2 <= len(w) <= 20]
        latin = sum(len(w) for w in words)
        if japanese and japanese * 2 >= latin:
            return "ja"
        return "en" if latin else None
    return None


REFUSED_NOTICE = "This request was refused.\nこのリクエストは拒否されました。"


def _refused_before_answering(state: Any) -> bool:
    """True when the S-1 trust gate refused the caller, so there is no answer to dress.

    Every other failure gets a trailer and a sentence a reader can act on. This one gets
    the refusal and nothing else -- no scope line, no guidance. A caller who is not
    permitted to invoke the agent must not be told what it is for, and `output: None` is
    equally wrong: the platform renders it as a blank screen under "agent failed", so the
    reader is told nothing, not even that they were refused. Two repo tests pulled in
    opposite directions on this and both were right about half of it.

    Read from `error_log`, which BaseNode.__call__ writes -- framework behaviour, the
    same in every repo, rather than a per-agent convention that would drift.
    """
    if not isinstance(state, dict):
        return False
    return any("S-1 trust gate denied" in str(e) for e in (state.get("error_log") or ()))


def _with_disclaimer(envelope: dict[str, Any], state: Any) -> dict[str, Any]:
    """Delegate to the shared envelope: one door for the whole fleet.

    This used to be a private copy. Keeping it private meant every change to the reply
    contract cost a refactor here instead of a file copy, and it silently missed the
    improvements the shared one gained -- the language decision, the single-closing-
    sentence rule, the blob check.

    Only the scope wording stays local, because only a reader of THIS agent can write it.
    """
    from src.services.agent_scope import SCOPE_EN, SCOPE_JA
    from src.services.output_envelope import with_disclaimer

    return with_disclaimer(envelope, state, scope_en=SCOPE_EN, scope_ja=SCOPE_JA)
