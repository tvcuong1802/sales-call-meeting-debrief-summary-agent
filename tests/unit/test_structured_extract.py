"""
Unit tests for StructuredExtractNode — CMN-C1-042.

Part 1 coverage:
- Happy path: BANT-complete transcript → all fields extracted
- Null fields: partial transcript → null for unmentioned fields (anti-hallucination)
- Markdown stripping: LLM wraps response in ```json``` → still parses correctly
- Error states: invalid JSON → LLM_SCHEMA_FAILURE error code
- Error states: Pydantic validation failure → LLM_SCHEMA_FAILURE error code
- Upstream error short-circuit
- No LLM client → LLM_SCHEMA_FAILURE error code
- S-4 trace: domain events emitted, no backbone node_start/complete/error

Part 2 coverage:
- MEDDIC methodology: fields in block, extraction, null fields, single LLM call
- Email formality branches: formal / semi_formal / casual blocks differ and inject correctly
- HubSpot schema: fields in prompt, not Salesforce fields, extraction happy path
- Config loading: override selects methodology/formality, missing file graceful
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.nodes.structured_extract import (
    StructuredExtractNode,
    DebriefOutput,
    _build_prompt,
    _build_crm_fields_block,
    _build_output_schema_block,
)
from src.schemas.state import initial_state
from tests.fixtures.transcripts import (
    BANT_COMPLETE_TRANSCRIPT,
    BANT_COMPLETE_LLM_RESPONSE,
    BANT_PARTIAL_TRANSCRIPT,
    BANT_PARTIAL_LLM_RESPONSE,
    MARKDOWN_WRAPPED_LLM_RESPONSE,
    INVALID_JSON_LLM_RESPONSE,
    MISSING_FIELD_LLM_RESPONSE,
)


# ── Shared helpers ────────────────────────────────────────────────────────────


def _make_state(annotated_transcript: str = "", **kwargs) -> dict:
    s = dict(initial_state(raw_transcript="test", session_id="test-session"))
    s["annotated_transcript"] = annotated_transcript
    s["invocation_id"] = "test-inv-001"
    s.update(kwargs)
    return s


def _cfg(**overrides):
    c = {
        "agent": {"output_language": "ja"},
        "methodology": "bant",
        "email": {"formality_level": "formal"},
        "crm": {"schema_path": "config/schemas/salesforce_opportunity.json"},
        "llm": {"model": "gpt-4o", "max_tokens": 2048, "temperature": 0.0},
        "nuance": {"dictionary_path": "config/nuance_ja.yaml"},
    }
    c.update(overrides)
    return c


def _make_node(response: str | None = None, config_override=None) -> tuple[StructuredExtractNode, MagicMock]:
    mock_client = MagicMock()
    if response is not None:
        mock_client.invoke.return_value = response
    cfg = config_override or _cfg()
    node = StructuredExtractNode(llm_client=mock_client, config_override=cfg)
    return node, mock_client


# ── Part 2 fixture responses ──────────────────────────────────────────────────

MEDDIC_LLM_RESPONSE = json.dumps(
    {
        "summary": "顧客は在庫管理コスト30%削減を目標。CFOが経済的バイヤー。導入は2Qを予定。",
        "crm_fields": {
            "Metrics": "在庫管理コスト30%削減",
            "EconomicBuyer": "CFO 田中様",
            "DecisionCriteria": "既存ERPとの統合可否、導入コスト",
            "DecisionProcess": "技術評価→CFO承認→契約（2Q）",
            "IdentifyPain": "手作業在庫管理によるミスと工数超過",
            "Champion": "情報システム部 鈴木様",
        },
        "email_draft": "Subject: 先日のご面談御礼\n\n田中様\n\nご面談ありがとうございました。",
    }
)

SEMI_FORMAL_LLM_RESPONSE = json.dumps(
    {
        "summary": "商談進捗の確認。予算感・スケジュール合意済み。",
        "crm_fields": {
            "Name": None,
            "StageName": "Qualification",
            "Amount": None,
            "NextStep": "来週再確認",
            "Description": None,
            "CloseDate": None,
        },
        "email_draft": "Subject: ご面談御礼\n\n先日はありがとうございました。",
    }
)

HUBSPOT_LLM_RESPONSE = json.dumps(
    {
        "summary": "HubSpot案件の確認。次ステップ合意。",
        "crm_fields": {
            "dealname": "Acme Integration Project",
            "dealstage": "appointmentscheduled",
            "amount": 500000,
            "closedate": "2026-07-31",
            "notes_last_contacted": "初回面談完了。統合要件を確認。",
            "hs_next_step": "技術デモ実施",
        },
        "email_draft": "Subject: ご面談御礼とデモのご案内\n\nAcme様\n\nありがとうございました。",
    }
)


# ════════════════════════════════════════════════════════════════════════════════
# PART 1 TESTS
# ════════════════════════════════════════════════════════════════════════════════

# ── DebriefOutput Pydantic model tests ────────────────────────────────────────


# These prompt-selection tests used to call execute() with an EMPTY transcript, and the
# node called the model anyway. It no longer does: a message carrying no letters carries
# no transcript, so there is nothing to summarise and no language to read. The tests are
# about which prompt block gets selected, so any real transcript serves -- an empty one
# was only ever a convenience.
_TRANSCRIPT_FOR_PROMPT_TESTS = "Call with Acme Corp. Budget approved. Next step: migration plan."


class TestDebriefOutput:
    def test_valid_construction(self):
        d = DebriefOutput(
            summary="test summary",
            crm_fields={"Name": "Deal A", "StageName": None},
            email_draft="Subject: Test\n\nBody",
        )
        assert d.summary == "test summary"
        assert d.crm_fields["StageName"] is None

    def test_missing_field_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DebriefOutput(summary="test", email_draft="test")  # missing crm_fields


# ── Prompt builder tests ──────────────────────────────────────────────────────


class TestBuildPrompt:
    def test_contains_role_block(self):
        node, _ = _make_node(BANT_COMPLETE_LLM_RESPONSE)
        prompt = _build_prompt(
            "sample transcript",
            methodology_block=node._methodology_block,
            email_block=node._email_block,
            crm_fields_block=node._crm_fields_block,
            output_schema_block=node._output_schema_block,
        )
        assert "structured extraction engine" in prompt

    def test_contains_anti_hallucination(self):
        node, _ = _make_node(BANT_COMPLETE_LLM_RESPONSE)
        prompt = _build_prompt(
            "sample transcript",
            methodology_block=node._methodology_block,
            email_block=node._email_block,
            crm_fields_block=node._crm_fields_block,
            output_schema_block=node._output_schema_block,
        )
        assert "null" in prompt
        assert "CRITICAL RULE" in prompt

    def test_contains_bant_fields(self):
        node, _ = _make_node(BANT_COMPLETE_LLM_RESPONSE)
        prompt = _build_prompt(
            "sample transcript",
            methodology_block=node._methodology_block,
            email_block=node._email_block,
            crm_fields_block=node._crm_fields_block,
            output_schema_block=node._output_schema_block,
        )
        assert "BANT" in prompt
        assert "StageName" in prompt

    def test_instruction_first_ordering(self):
        node, _ = _make_node(BANT_COMPLETE_LLM_RESPONSE)
        prompt = _build_prompt(
            "transcript content",
            methodology_block=node._methodology_block,
            email_block=node._email_block,
            crm_fields_block=node._crm_fields_block,
            output_schema_block=node._output_schema_block,
        )
        role_pos = prompt.find("structured extraction engine")
        transcript_pos = prompt.find("transcript content")
        assert role_pos < transcript_pos

    def test_transcript_delimited(self):
        node, _ = _make_node(BANT_COMPLETE_LLM_RESPONSE)
        prompt = _build_prompt(
            "transcript content",
            methodology_block=node._methodology_block,
            email_block=node._email_block,
            crm_fields_block=node._crm_fields_block,
            output_schema_block=node._output_schema_block,
        )
        assert "--- TRANSCRIPT BEGIN ---" in prompt
        assert "--- TRANSCRIPT END ---" in prompt


# ── Happy path tests ──────────────────────────────────────────────────────────


class TestStructuredExtractHappyPath:
    def test_bant_complete_all_fields_extracted(self):
        node, _ = _make_node(BANT_COMPLETE_LLM_RESPONSE)
        state = _make_state(annotated_transcript=BANT_COMPLETE_TRANSCRIPT)
        result = node.execute(state)
        assert result["error"] == ""
        assert result["summary"] != ""
        crm = json.loads(result["crm_fields"])
        assert crm["Amount"] == 3000000
        assert crm["StageName"] == "Needs Analysis"
        assert result["email_draft"] != ""

    def test_crm_fields_stored_as_json_string(self):
        node, _ = _make_node(BANT_COMPLETE_LLM_RESPONSE)
        state = _make_state(annotated_transcript=BANT_COMPLETE_TRANSCRIPT)
        result = node.execute(state)
        assert isinstance(result["crm_fields"], str)
        json.loads(result["crm_fields"])

    def test_bant_partial_null_fields_respected(self):
        node, _ = _make_node(BANT_PARTIAL_LLM_RESPONSE)
        state = _make_state(annotated_transcript=BANT_PARTIAL_TRANSCRIPT)
        result = node.execute(state)
        crm = json.loads(result["crm_fields"])
        assert crm["Amount"] is None
        assert crm["CloseDate"] is None
        assert crm["NextStep"] is None

    def test_llm_invoked_exactly_once(self):
        node, mock_client = _make_node(BANT_COMPLETE_LLM_RESPONSE)
        state = _make_state(annotated_transcript=BANT_COMPLETE_TRANSCRIPT)
        node.execute(state)
        mock_client.invoke.assert_called_once()

    def test_prompt_passed_to_llm(self):
        node, mock_client = _make_node(BANT_COMPLETE_LLM_RESPONSE)
        state = _make_state(annotated_transcript="unique test transcript content xyz")
        node.execute(state)
        prompt_arg = mock_client.invoke.call_args[0][0]
        assert "unique test transcript content xyz" in prompt_arg
        assert "CRITICAL RULE" in prompt_arg


# ── Markdown stripping tests ──────────────────────────────────────────────────


class TestMarkdownStripping:
    def test_json_code_fence_stripped(self):
        node, _ = _make_node(MARKDOWN_WRAPPED_LLM_RESPONSE)
        state = _make_state(annotated_transcript="テスト")
        result = node.execute(state)
        assert result["error"] == ""
        assert result["summary"] == "テスト用サマリー"

    def test_parse_response_strips_fence(self):
        node, _ = _make_node()
        debrief = node._parse_response(MARKDOWN_WRAPPED_LLM_RESPONSE)
        assert isinstance(debrief, DebriefOutput)
        assert debrief.crm_fields["Name"] == "テスト案件"


# ── Error state tests ─────────────────────────────────────────────────────────


class TestErrorStates:
    def test_invalid_json_sets_schema_failure(self):
        node, _ = _make_node(INVALID_JSON_LLM_RESPONSE)
        result = node.execute(_make_state(annotated_transcript="test"))
        err = json.loads(result["error"])
        assert err["code"] == "LLM_SCHEMA_FAILURE"
        assert err["node"] == "StructuredExtractNode"

    def test_missing_pydantic_field_sets_schema_failure(self):
        node, _ = _make_node(MISSING_FIELD_LLM_RESPONSE)
        result = node.execute(_make_state(annotated_transcript="test"))
        err = json.loads(result["error"])
        assert err["code"] == "LLM_SCHEMA_FAILURE"

    def test_upstream_error_short_circuits(self):
        node, mock_client = _make_node()
        state = _make_state(
            error=json.dumps({"code": "S1_INPUT_REJECTED", "attempt": 1, "node": "TranscriptCleanNode"})
        )
        result = node.execute(state)
        mock_client.invoke.assert_not_called()
        err = json.loads(result["error"])
        assert err["code"] == "S1_INPUT_REJECTED"

    def test_no_llm_client_fails_closed_to_empty_advisory(self):
        # No LLM → fail-closed to an empty advisory (cannot_summarize), NOT an error
        # (wheel/STG contract: the pipeline must complete successfully with no LLM).
        node = StructuredExtractNode(llm_client=None, config_override=_cfg())
        result = node.execute(_make_state(annotated_transcript="test"))
        assert not result.get("error")
        assert result["cannot_summarize"] is True
        assert result["summary"] == ""

    def test_llm_raises_unexpected_exception(self):
        node, mock_client = _make_node()
        mock_client.invoke.side_effect = RuntimeError("unexpected")
        result = node.execute(_make_state(annotated_transcript="test"))
        err = json.loads(result["error"])
        assert err["code"] == "LLM_SCHEMA_FAILURE"


# ── S-4 trace event tests ─────────────────────────────────────────────────────

# Domain nodes MUST NOT emit framework backbone lifecycle events — those are
# owned by BaseNode.__call__; duplicating them corrupts the audit trail
# (CoE Criterion #2).
_BACKBONE = {"node_start", "node_complete", "node_error", "node_skip"}


class TestS4TraceEvents:
    def test_domain_event_emitted_on_success(self):
        node, _ = _make_node(BANT_COMPLETE_LLM_RESPONSE)
        with patch(type(node).__module__ + ".emit_trace_event") as mock_emit:
            node.execute(_make_state(annotated_transcript=BANT_COMPLETE_TRANSCRIPT))
        events = [c[0][0] for c in mock_emit.call_args_list]
        assert "structured_extracted" in events
        assert not (_BACKBONE & set(events))

    def test_domain_error_emitted_on_parse_failure(self):
        node, _ = _make_node(INVALID_JSON_LLM_RESPONSE)
        with patch(type(node).__module__ + ".emit_trace_event") as mock_emit:
            node.execute(_make_state(annotated_transcript="test"))
        events = [c[0][0] for c in mock_emit.call_args_list]
        assert "structured_extract_error" in events
        assert not (_BACKBONE & set(events))

    def test_no_backbone_event_emitted(self):
        node, _ = _make_node(BANT_COMPLETE_LLM_RESPONSE)
        with patch(type(node).__module__ + ".emit_trace_event") as mock_emit:
            node.execute(_make_state(annotated_transcript="test", invocation_id="inv-trace-test"))
        events = [c[0][0] for c in mock_emit.call_args_list]
        assert events
        assert not (_BACKBONE & set(events))

    def test_success_event_carries_invocation_id(self):
        node, _ = _make_node(BANT_COMPLETE_LLM_RESPONSE)
        with patch(type(node).__module__ + ".emit_trace_event") as mock_emit:
            node.execute(_make_state(annotated_transcript=BANT_COMPLETE_TRANSCRIPT, invocation_id="inv-trace-test"))
        done = [c for c in mock_emit.call_args_list if c[0][0] == "structured_extracted"]
        assert done[0][0][1]["invocation_id"] == "inv-trace-test"


# ════════════════════════════════════════════════════════════════════════════════
# PART 2 TESTS — dynamic injection: MEDDIC, email formality, HubSpot schema
# ════════════════════════════════════════════════════════════════════════════════


class TestBuildCrmFieldsBlock:
    def test_salesforce_fields_appear(self):
        schema = json.load(open("config/schemas/salesforce_opportunity.json"))
        block = _build_crm_fields_block(schema)
        assert "Name" in block
        assert "StageName" in block
        assert "Amount" in block
        assert "CloseDate" in block

    def test_hubspot_fields_appear(self):
        schema = json.load(open("config/schemas/hubspot_deal.json"))
        block = _build_crm_fields_block(schema)
        assert "dealname" in block
        assert "hs_next_step" in block

    def test_empty_schema_returns_generic(self):
        block = _build_crm_fields_block({})
        assert block != ""

    def test_output_schema_block_salesforce(self):
        schema = json.load(open("config/schemas/salesforce_opportunity.json"))
        block = _build_output_schema_block(schema)
        assert "Name" in block and "StageName" in block

    def test_output_schema_block_hubspot(self):
        schema = json.load(open("config/schemas/hubspot_deal.json"))
        block = _build_output_schema_block(schema)
        assert "dealname" in block and "hs_next_step" in block


class TestMEDDICMethodology:
    def _node(self, resp=None):
        mc = MagicMock()
        mc.invoke.return_value = resp or MEDDIC_LLM_RESPONSE
        return StructuredExtractNode(llm_client=mc, config_override=_cfg(methodology="meddic"))

    def test_meddic_fields_in_methodology_block(self):
        n = self._node()
        for field in ["Metrics", "EconomicBuyer", "IdentifyPain", "Champion"]:
            assert field in n._methodology_block

    def test_bant_fields_not_in_meddic_block(self):
        n = self._node()
        assert "StageName" not in n._methodology_block
        assert "CloseDate" not in n._methodology_block

    def test_meddic_extraction_happy_path(self):
        n = self._node()
        result = n.execute(_make_state(annotated_transcript=_TRANSCRIPT_FOR_PROMPT_TESTS))
        assert result["error"] == ""
        crm = json.loads(result["crm_fields"])
        assert crm["Metrics"] == "在庫管理コスト30%削減"
        assert crm["Champion"] == "情報システム部 鈴木様"

    def test_meddic_null_fields_respected(self):
        resp = json.dumps(
            {
                "summary": "初期接触。",
                "crm_fields": {
                    "Metrics": None,
                    "EconomicBuyer": None,
                    "DecisionCriteria": None,
                    "DecisionProcess": None,
                    "IdentifyPain": "手作業非効率",
                    "Champion": None,
                },
                "email_draft": "Subject: 御礼\n\n本文",
            }
        )
        n = self._node(resp)
        result = n.execute(_make_state(annotated_transcript=_TRANSCRIPT_FOR_PROMPT_TESTS))
        crm = json.loads(result["crm_fields"])
        assert crm["Metrics"] is None
        assert crm["IdentifyPain"] == "手作業非効率"

    def test_meddic_single_llm_call(self):
        n = self._node()
        n.execute(_make_state(annotated_transcript=_TRANSCRIPT_FOR_PROMPT_TESTS))
        n._llm_client.invoke.assert_called_once()


class TestEmailFormality:
    def _node(self, formality, resp=None):
        mc = MagicMock()
        mc.invoke.return_value = resp or SEMI_FORMAL_LLM_RESPONSE
        return StructuredExtractNode(
            llm_client=mc,
            config_override=_cfg(**{"email": {"formality_level": formality}}),
        )

    def test_formal_block_loaded(self):
        n = self._node("formal")
        assert n._email_block.strip() != ""
        assert "敬語" in n._email_block or "丁寧語" in n._email_block

    def test_semi_formal_block_loaded(self):
        n = self._node("semi_formal")
        assert "丁寧語" in n._email_block

    def test_casual_block_loaded(self):
        n = self._node("casual")
        assert "平語" in n._email_block or "カジュアル" in n._email_block

    def test_formal_and_casual_differ(self):
        assert self._node("formal")._email_block != self._node("casual")._email_block

    def test_semi_formal_and_formal_differ(self):
        assert self._node("formal")._email_block != self._node("semi_formal")._email_block

    def test_formality_injected_into_prompt(self):
        n = self._node("casual")
        n.execute(_make_state(annotated_transcript="test"))
        prompt = n._llm_client.invoke.call_args[0][0]
        assert n._email_block.strip()[:20] in prompt


class TestHubSpotSchema:
    def _node(self, resp=None):
        mc = MagicMock()
        mc.invoke.return_value = resp or HUBSPOT_LLM_RESPONSE
        return StructuredExtractNode(
            llm_client=mc,
            config_override=_cfg(**{"crm": {"schema_path": "config/schemas/hubspot_deal.json"}}),
        )

    def test_hubspot_fields_in_crm_block(self):
        n = self._node()
        assert "dealname" in n._crm_fields_block
        assert "hs_next_step" in n._crm_fields_block

    def test_salesforce_fields_not_in_hubspot_prompt(self):
        n = self._node()
        assert "StageName" not in n._crm_fields_block
        assert "StageName" not in n._output_schema_block

    def test_hubspot_extraction_happy_path(self):
        n = self._node()
        result = n.execute(_make_state(annotated_transcript=_TRANSCRIPT_FOR_PROMPT_TESTS))
        assert result["error"] == ""
        crm = json.loads(result["crm_fields"])
        assert crm["dealname"] == "Acme Integration Project"
        assert crm["hs_next_step"] == "技術デモ実施"
        assert crm["amount"] == 500000

    def test_hubspot_single_llm_call(self):
        n = self._node()
        n.execute(_make_state(annotated_transcript=_TRANSCRIPT_FOR_PROMPT_TESTS))
        n._llm_client.invoke.assert_called_once()


class TestConfigLoadingP2:
    def test_config_override_selects_meddic(self):
        mc = MagicMock()
        mc.invoke.return_value = MEDDIC_LLM_RESPONSE
        n = StructuredExtractNode(llm_client=mc, config_override=_cfg(methodology="meddic"))
        assert "Metrics" in n._methodology_block

    def test_config_override_selects_casual(self):
        mc = MagicMock()
        mc.invoke.return_value = SEMI_FORMAL_LLM_RESPONSE
        n = StructuredExtractNode(
            llm_client=mc,
            config_override=_cfg(**{"email": {"formality_level": "casual"}}),
        )
        assert "平語" in n._email_block or "カジュアル" in n._email_block

    def test_missing_config_file_graceful(self, tmp_path):
        mc = MagicMock()
        mc.invoke.return_value = MEDDIC_LLM_RESPONSE
        n = StructuredExtractNode(
            llm_client=mc,
            config_path=str(tmp_path / "nonexistent.yaml"),
        )
        assert n._methodology_block != ""


# ── TestRetryWiring (Part 3) ──────────────────────────────────────────────────


class TestRetryWiring:
    """Part 3: verify call_with_retry_typed() is wired into StructuredExtractNode.execute().

    Tests:
    - Retry fires on LLM_SCHEMA_FAILURE (bad JSON on attempt 1, good JSON on attempt 2)
    - node_retry trace event emitted per retry attempt
    - Exhausted retries sets state error correctly
    """

    def _make_retry_node(self, responses: list, config_override=None) -> tuple[StructuredExtractNode, MagicMock]:
        """Build a node whose LLM client returns successive responses from the list."""
        mock_client = MagicMock()
        mock_client.invoke.side_effect = responses
        cfg = config_override or _cfg()
        node = StructuredExtractNode(llm_client=mock_client, config_override=cfg)
        return node, mock_client

    def test_retry_fires_on_bad_json_then_good(self):
        """Bad JSON on attempt 1 → ValidationRetryError → retry → good JSON succeeds."""
        node, mock_client = self._make_retry_node(
            [
                INVALID_JSON_LLM_RESPONSE,  # attempt 1: bad JSON → ValidationRetryError
                BANT_COMPLETE_LLM_RESPONSE,  # attempt 2 (retry): good JSON
            ]
        )
        state = _make_state(annotated_transcript=BANT_COMPLETE_TRANSCRIPT)
        result = node.execute(state)
        # Should succeed on retry
        assert result["error"] == ""
        assert result["summary"] != ""
        crm = json.loads(result["crm_fields"])
        assert crm["Amount"] == 3000000
        # LLM was called twice (attempt 1 + retry)
        assert mock_client.invoke.call_count == 2

    def test_node_retry_trace_event_emitted_on_retry(self):
        """node_retry trace event must be emitted when retry fires."""
        node, _ = self._make_retry_node(
            [
                INVALID_JSON_LLM_RESPONSE,
                BANT_COMPLETE_LLM_RESPONSE,
            ]
        )
        state = _make_state(annotated_transcript=BANT_COMPLETE_TRANSCRIPT)
        with patch(type(node).__module__ + ".emit_trace_event") as mock_emit:
            node.execute(state)
        events = [c[0][0] for c in mock_emit.call_args_list]
        assert "node_retry" in events, f"node_retry not in emitted events: {events}"

    def test_node_retry_event_has_correct_fields(self):
        """node_retry event must carry attempt number, error_type, node_name."""
        node, _ = self._make_retry_node(
            [
                INVALID_JSON_LLM_RESPONSE,
                BANT_COMPLETE_LLM_RESPONSE,
            ]
        )
        state = _make_state(annotated_transcript=BANT_COMPLETE_TRANSCRIPT)
        with patch(type(node).__module__ + ".emit_trace_event") as mock_emit:
            node.execute(state)
        retry_calls = [c for c in mock_emit.call_args_list if c[0][0] == "node_retry"]
        assert len(retry_calls) >= 1
        payload = retry_calls[0][0][1]
        assert "attempt" in payload
        assert "error_type" in payload
        assert "node" in payload
        assert payload["attempt"] == 1
        assert payload["error_type"] == "LLM_SCHEMA_FAILURE"
        assert payload["node"] == "StructuredExtractNode"

    def test_exhausted_retries_sets_state_error(self):
        """When all retry attempts fail, state["error"] must be set with LLM_SCHEMA_FAILURE."""
        node, mock_client = self._make_retry_node(
            [
                INVALID_JSON_LLM_RESPONSE,  # attempt 1 → ValidationRetryError
                INVALID_JSON_LLM_RESPONSE,  # attempt 2 (retry) → still bad → exhausted
            ]
        )
        state = _make_state(annotated_transcript="test transcript")
        result = node.execute(state)
        assert result["error"] != ""
        err = json.loads(result["error"])
        assert err["code"] == "LLM_SCHEMA_FAILURE"
        assert err["node"] == "StructuredExtractNode"

    def test_adr_001_still_holds_on_success_path(self):
        """ADR-001: on success path (no retry needed), exactly 1 LLM call."""
        node, mock_client = self._make_retry_node([BANT_COMPLETE_LLM_RESPONSE])
        state = _make_state(annotated_transcript=BANT_COMPLETE_TRANSCRIPT)
        node.execute(state)
        mock_client.invoke.assert_called_once()

    def test_domain_success_event_fires_correctly(self):
        """Success domain event must fire (and no error event) even with retry wiring."""
        node, _ = self._make_retry_node([BANT_COMPLETE_LLM_RESPONSE])
        state = _make_state(annotated_transcript=BANT_COMPLETE_TRANSCRIPT)
        with patch(type(node).__module__ + ".emit_trace_event") as mock_emit:
            node.execute(state)
        events = [c[0][0] for c in mock_emit.call_args_list]
        assert "structured_extracted" in events
        assert "structured_extract_error" not in events
        assert not (_BACKBONE & set(events))

    def test_domain_error_fires_on_exhausted_retry(self):
        """Domain error event must fire when all retries are exhausted."""
        node, _ = self._make_retry_node(
            [
                INVALID_JSON_LLM_RESPONSE,
                INVALID_JSON_LLM_RESPONSE,
            ]
        )
        state = _make_state(annotated_transcript="test")
        with patch(type(node).__module__ + ".emit_trace_event") as mock_emit:
            node.execute(state)
        events = [c[0][0] for c in mock_emit.call_args_list]
        assert "structured_extract_error" in events
        assert not (_BACKBONE & set(events))

    def test_retry_config_from_agent_config(self):
        """Retry reads max_attempts from config.yaml llm.retry.max_attempts."""
        # With max_attempts=1, second failure should immediately set error
        cfg = _cfg()
        cfg["llm"] = {
            "model": "gpt-4o",
            "max_tokens": 2048,
            "temperature": 0.0,
            "retry": {"max_attempts": 1, "backoff_factor": 0.0},
        }
        node, mock_client = self._make_retry_node(
            [INVALID_JSON_LLM_RESPONSE, INVALID_JSON_LLM_RESPONSE],
            config_override=cfg,
        )
        state = _make_state(annotated_transcript="test")
        result = node.execute(state)
        err = json.loads(result["error"])
        assert err["code"] == "LLM_SCHEMA_FAILURE"
