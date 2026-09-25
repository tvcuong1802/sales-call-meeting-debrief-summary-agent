"""
tests/unit/test_graph.py — CMN-C1-042 new-gen graph composition + slot pipeline

New-gen: Graph(AgentBaseGraph) fills three backbone slots (pre_process / main /
post_process) with orchestrator FunctionNodes that fold the five domain sub-nodes
(DI + execute(), no GraphNode). The framework owns compile()/invoke()/routing and
enforces S-1 (required_trust_level) in BaseNode.__call__. There is no agent-level
run()/_invoke_impl()/_security_gate_*.
"""

from __future__ import annotations

import ast
import json
import pathlib

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.schemas.invocation_context import TrustLevel

from src.graph.graph import Graph
from src.nodes.pre_process_node import PreProcessNode
from src.nodes.main_node import MainNode
from src.nodes.post_process_node import PostProcessNode
from src.schemas.state import initial_state


def _run_full(state: dict) -> dict:
    # Nodes return partial-update dicts (wheel contract); merge like the framework.
    state = {**state, **PreProcessNode().execute(state)}
    state = {**state, **MainNode().execute(state)}
    state = {**state, **PostProcessNode().execute(state)}
    return state


# ── Composition ────────────────────────────────────────────────────────────────


class TestGraphComposition:
    EXPECTED = {"pre_process": PreProcessNode, "main": MainNode, "post_process": PostProcessNode}

    def test_three_slots_registered(self):
        g = Graph()
        g.register_nodes()
        for slot, cls in self.EXPECTED.items():
            assert slot in g._nodes and isinstance(g._nodes[slot], cls)

    def test_name_property(self):
        assert Graph().name == "cmn_c1_042"

    def test_inherits_agent_base_graph(self):
        assert issubclass(Graph, AgentBaseGraph)

    def test_no_level0_import(self):
        tree = ast.parse(pathlib.Path("src/graph/graph.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = node.module if isinstance(node, ast.ImportFrom) else "".join(a.name for a in node.names)
                assert "agenticstar" not in (mod or "")


# ── S-1 trust level ─────────────────────────────────────────────────────────────


class TestTrustLevel:
    def test_orchestrators_declare_verified_external(self):
        for cls in (PreProcessNode, MainNode, PostProcessNode):
            assert cls.required_trust_level == TrustLevel.VERIFIED_EXTERNAL
            assert "required_trust_level" in cls.__dict__


# ── pre_process slot (transcript clean S-2 + nuance) ────────────────────────────


class TestPreProcessSlot:
    def test_normal_transcript_cleaned(self):
        _s = initial_state("Sales call with Acme. Demo next week.", "s1")
        result = {**_s, **PreProcessNode().execute(_s)}
        assert result["cleaned_transcript"]
        assert result.get("invocation_id")  # set by TranscriptCleanNode

    def test_empty_transcript_sets_s1_error(self):
        result = PreProcessNode().execute(initial_state("   ", "s2"))
        assert result["error"]
        assert "S1_INPUT_REJECTED" in result["error"]


# ── main slot (extract + crm validate) ──────────────────────────────────────────


class TestMainSlot:
    def test_error_state_short_circuits(self):
        state = initial_state("x", "s3")
        state["error"] = json.dumps({"code": "UPSTREAM", "attempt": 1})
        result = MainNode().execute(state)
        assert result["summary"] == ""  # extraction did not run

    def test_llm_none_fails_closed_to_success(self):
        # No LLM injected → fail-closed to an empty advisory + SUCCESS (never error).
        state = {
            **initial_state("Sales call. Budget approved.", "s4"),
            **PreProcessNode().execute(initial_state("Sales call. Budget approved.", "s4")),
        }
        result = MainNode().execute(state)
        assert not result.get("error")
        assert result["cannot_summarize"] is True


# ── post_process slot (S-3 output gate, non-suppressible) ───────────────────────


class TestPostProcessSlot:
    def test_s3_runs_even_with_error(self):
        state = initial_state("x", "s5")
        state["error"] = json.dumps({"code": "UPSTREAM", "attempt": 1, "node": "n"})
        result = PostProcessNode().execute(state)
        # OutputFormatNode (S-3) always runs → output_json + audit_output_hash populated.
        assert result["output_json"]
        assert result["audit_output_hash"]

    def test_clean_state_formats_output(self):
        state = initial_state("x", "s6")
        state["summary"] = "Met with Acme; demo scheduled."
        result = PostProcessNode().execute(state)
        assert result["output_json"]


# ── end-to-end orchestrator chain ───────────────────────────────────────────────


class TestEndToEnd:
    def test_empty_transcript_halts_with_structured_error_s3_runs(self):
        result = _run_full(initial_state("   ", "e1"))
        assert result["error"] and "S1_INPUT_REJECTED" in result["error"]
        assert result["output_json"]  # S-3 / OutputFormat still ran
        assert result["audit_output_hash"]

    def test_no_llm_pipeline_completes_success_with_output(self):
        result = _run_full(initial_state("Sales call with Acme. Demo next week. Budget approved.", "e2"))
        assert not result.get("error")  # fail-closed, not error
        assert result.get("cannot_summarize") is True
        assert result["output_json"]  # S-3 still produced an output envelope
        json.dumps(result)  # state is JSON/msgpack-safe
