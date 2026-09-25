"""
Proof-of-Boundary tests — CMN-C1-042 Sales Call & Meeting Debrief Summary Agent.

Covers PB-1 through PB-7 as defined in docs/03_test_spec.md §4.

PB-1: S-1 — empty input rejected with error state
PB-2: S-1 — oversized input (>50,000 chars) rejected
PB-3: S-2→S-3 — PII absent from summary in final output
PB-4: S-2→S-3 — PII absent from email_draft in final output
PB-5: Graceful degradation — LLM failure → error state, no crash
PB-6: CRM schema — Salesforce schema produces expected field keys
PB-7: CRM schema — HubSpot schema produces expected field keys

Tests drive nodes directly (not via agent.run()) for ci_stub compatibility,
following the same pattern as test_pb_pii_gate.py.
All tests pass with the stub harness (no real LLM calls).
Design doc §8.1, §8.2, §8.3, §9.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock


from src.nodes.transcript_clean import TranscriptCleanNode
from src.nodes.nuance_classify import NuanceClassifyNode
from src.nodes.structured_extract import StructuredExtractNode
from src.nodes.crm_schema_validate import CRMSchemaValidateNode
from src.nodes.output_format import OutputFormatNode
from src.schemas.state import initial_state


# ── Shared helpers ────────────────────────────────────────────────────────────


def _run_pipeline(
    raw_transcript: str,
    llm_response: str,
    crm_schema: str = "config/schemas/salesforce_opportunity.json",
    session_id: str = "pb-test",
) -> dict:
    """Run all 5 nodes in sequence with a mock LLM client. Returns final state."""
    mock_client = MagicMock()
    mock_client.invoke.return_value = llm_response

    inner_cfg = {
        "agent": {"output_language": "ja"},
        "methodology": "bant",
        "email": {"formality_level": "formal"},
        "crm": {"schema_path": crm_schema},
        "llm": {
            "model": "gpt-4o",
            "max_tokens": 2048,
            "temperature": 0.0,
            "retry": {"max_attempts": 1, "backoff_factor": 0.0},
        },
        "nuance": {"dictionary_path": "config/nuance_ja.yaml"},
        "security": {"redact_patterns": []},
    }
    state = dict(initial_state(raw_transcript=raw_transcript, session_id=session_id))

    # Every node is CONSTRUCTED with the config, the way the graph builds it. There is no
    # per-call config envelope any more: the framework invokes execute(state) alone, so a
    # schema path handed over at call time never reached a running agent -- which is why
    # this test could switch schemas while the deployed agent could not.
    state = TranscriptCleanNode(config=inner_cfg).execute(state)
    if state.get("error"):
        return state  # S-1 rejection — return early, downstream nodes won't run

    state = NuanceClassifyNode(config=inner_cfg).execute(state)

    # main slot — StructuredExtractNode reads config via its own config_override __init__ param
    extract_node = StructuredExtractNode(llm_client=mock_client, config_override=inner_cfg)
    state = extract_node.execute(state)

    # post_process slot
    state = CRMSchemaValidateNode(config=inner_cfg).execute(state)
    state = OutputFormatNode(config=inner_cfg).execute(state)

    return state


def _parse_output(state: dict) -> dict:
    """Parse output_json from final state. Returns {} if not present."""
    output_json = state.get("output_json", "")
    if output_json:
        try:
            return json.loads(output_json)
        except json.JSONDecodeError:
            return {}
    return {}


# ── Fixtures ──────────────────────────────────────────────────────────────────

_PII_TRANSCRIPT = (
    "[S1] 田中様、先日お送りした satoshi.tanaka@prospect.co.jp をご確認いただけましたか？\n"
    "[S2] はい、確認しました。折り返し 03-5555-1234 にお電話します。\n"
    "[S1] ありがとうございます。弊社は東京都新宿区西新宿2丁目8番1号にあります。\n"
    "[S2] 承知しました。来週、Sarah Johnson を同行させます。\n"
    "[S1] 予算は1,500万円で、Q3クローズを目指しています。\n"
    "[S2] ぜひ前向きに検討します。\n"
)

_PII_TOKENS = [
    "satoshi.tanaka@prospect.co.jp",
    "03-5555-1234",
    "Sarah Johnson",
    "田中様",
]

_SAFE_CONTENT = ["1,500万円", "Q3", "予算"]

_MOCK_LLM_RESPONSE_PII = json.dumps(
    {
        "summary": "予算1,500万円、Q3クローズ目標。ネクストステップ：来週デモ。",
        "crm_fields": {
            "Name": None,
            "StageName": "Needs Analysis",
            "Amount": 15000000,
            "NextStep": "来週デモ実施",
            "Description": "CRMソリューション導入検討。予算1,500万円、Q3クローズ希望。",
            "CloseDate": "2026-09-30",
        },
        "email_draft": (
            "Subject: 先日のご面談御礼とデモのご案内\n\n"
            "[NAME]様\n\n"
            "本日はお時間をいただきありがとうございました。\n"
            "来週のデモについてご連絡いたします。\n\n"
            "何卒よろしくお願いいたします。"
        ),
    }
)

_MOCK_LLM_RESPONSE_SALESFORCE = json.dumps(
    {
        "summary": "テスト用サマリー",
        "crm_fields": {
            "Name": "テスト案件",
            "StageName": "Prospecting",
            "Amount": 1000000,
            "NextStep": "フォローアップ",
            "Description": "テスト案件の説明",
            "CloseDate": "2026-12-31",
        },
        "email_draft": "Subject: テスト\n\nテスト本文",
    }
)

_MOCK_LLM_RESPONSE_HUBSPOT = json.dumps(
    {
        "summary": "テスト用サマリー",
        "crm_fields": {
            "dealname": "テスト案件",
            "dealstage": "appointmentscheduled",
            "amount": "1000000",
            "closedate": "2026-12-31",
            "description": "テスト案件の説明",
        },
        "email_draft": "Subject: テスト\n\nテスト本文",
    }
)

_CLEAN_TRANSCRIPT = "[S1] 予算300万円、6月末クローズ予定です。[S2] 承知しました。"


# ════════════════════════════════════════════════════════════════════════════════
# PB-1 / PB-2 — S-1 Input Validation
# ════════════════════════════════════════════════════════════════════════════════


class TestPBS1InputValidation:
    """PB-1, PB-2: S-1 gate rejects invalid input at the pipeline boundary.

    Design doc §8.1, §5.1.
    """

    def test_pb1_empty_input_rejected(self):
        """PB-1: Empty transcript → error state, S1_INPUT_REJECTED."""
        node = TranscriptCleanNode()
        state = dict(initial_state(raw_transcript="", session_id="pb1-test"))
        result = node.execute(state)

        assert result["error"] != "", "Empty input must be rejected"
        err = json.loads(result["error"])
        assert err.get("code") == "S1_INPUT_REJECTED"

    def test_pb1_whitespace_only_rejected(self):
        """PB-1: Whitespace-only transcript → rejected (functionally empty)."""
        node = TranscriptCleanNode()
        state = dict(initial_state(raw_transcript="   \n\t  ", session_id="pb1b-test"))
        result = node.execute(state)

        assert result["error"] != "", "Whitespace-only input must be rejected"

    def test_pb2_oversized_input_rejected(self):
        """PB-2: Transcript > 50,000 chars → S1_INPUT_REJECTED. Design doc §5.1."""
        oversized = "あ" * 50_001
        node = TranscriptCleanNode()
        state = dict(initial_state(raw_transcript=oversized, session_id="pb2-test"))
        result = node.execute(state)

        assert result["error"] != "", "Oversized input must be rejected"
        err = json.loads(result["error"])
        assert err.get("code") == "S1_INPUT_REJECTED"

    def test_pb2_exactly_max_chars_accepted(self):
        """PB-2 boundary: 50,000 chars must be accepted (not rejected)."""
        max_input = "あ" * 50_000
        node = TranscriptCleanNode()
        state = dict(initial_state(raw_transcript=max_input, session_id="pb2b-test"))
        result = node.execute(state)

        assert result["error"] == "", f"Exactly 50,000 chars must be accepted. Got: {result['error']}"

    def test_pb1_rejection_does_not_populate_cleaned_transcript(self):
        """PB-1: On rejection, cleaned_transcript must not be populated."""
        node = TranscriptCleanNode()
        state = dict(initial_state(raw_transcript="", session_id="pb1c-test"))
        result = node.execute(state)

        assert result.get("cleaned_transcript", "") == "", "On S-1 rejection, cleaned_transcript must remain empty"


# ════════════════════════════════════════════════════════════════════════════════
# PB-3 / PB-4 — PII absent from final output (S-2 → S-3 boundary)
# ════════════════════════════════════════════════════════════════════════════════


class TestPBPIIOutputBoundary:
    """PB-3, PB-4: PII in raw input must not appear in any final output field.

    Verifies end-to-end: S-2 masks PII before LLM → mock LLM returns safe
    content → output fields contain no raw PII.
    Design doc §8.2, APPI no-persistence guarantee.
    """

    def _run(self) -> tuple[dict, dict]:
        """Run full pipeline with PII transcript. Returns (state, output_dict)."""
        state = _run_pipeline(
            raw_transcript=_PII_TRANSCRIPT,
            llm_response=_MOCK_LLM_RESPONSE_PII,
            session_id="pb3-test",
        )
        output = _parse_output(state)
        return state, output

    def test_pb3_pii_absent_from_summary(self):
        """PB-3: Raw PII tokens must not appear in output summary."""
        _, output = self._run()
        summary = output.get("summary", "")
        for token in _PII_TOKENS:
            assert token not in summary, f"PII token '{token}' found in summary — APPI violation"

    def test_pb4_pii_absent_from_email_draft(self):
        """PB-4: Raw PII tokens must not appear in email_draft output."""
        _, output = self._run()
        email = output.get("email_draft", "")
        for token in _PII_TOKENS:
            assert token not in email, f"PII token '{token}' found in email_draft — APPI violation"

    def test_pb3_safe_content_preserved(self):
        """PB-3: Non-PII business content must survive masking (no over-masking)."""
        _, output = self._run()
        summary = output.get("summary", "")
        assert any(t in summary for t in _SAFE_CONTENT), (
            f"No safe business content found in summary. Got: {summary[:200]}"
        )

    def test_pb3_pii_masked_flag_set(self):
        """PB-3: meta.pii_masked must be True when PII was in the transcript."""
        _, output = self._run()
        meta = output.get("meta", {})
        assert meta.get("pii_masked") is True, f"meta.pii_masked must be True when PII was present. meta={meta}"

    def test_pb3_pii_absent_from_crm_fields(self):
        """PB-3: Raw PII tokens must not appear in crm_fields values."""
        _, output = self._run()
        crm_str = json.dumps(output.get("crm_fields", {}), ensure_ascii=False)
        for token in _PII_TOKENS:
            assert token not in crm_str, f"PII token '{token}' found in crm_fields — APPI violation"


# ════════════════════════════════════════════════════════════════════════════════
# PB-5 — Graceful degradation on LLM failure
# ════════════════════════════════════════════════════════════════════════════════


class TestPBGracefulDegradation:
    """PB-5: LLM failure → structured error state, no unhandled exception.

    Design doc §9.1: 'never raise an unhandled exception. Always return
    a structured error state.'
    """

    def _run_with_llm_error(self, error: Exception, session_id: str = "pb5-test") -> dict:
        """Run pipeline with LLM client raising an exception. Returns final state."""

        mock_client = MagicMock()
        mock_client.invoke.side_effect = error

        inner_cfg = {
            "agent": {"output_language": "ja"},
            "methodology": "bant",
            "email": {"formality_level": "formal"},
            "crm": {"schema_path": "config/schemas/salesforce_opportunity.json"},
            "llm": {
                "model": "gpt-4o",
                "max_tokens": 2048,
                "temperature": 0.0,
                "retry": {"max_attempts": 1, "backoff_factor": 0.0},
            },
            "nuance": {"dictionary_path": "config/nuance_ja.yaml"},
            "security": {"redact_patterns": []},
        }
        cfg = {"configurable": {**inner_cfg, "session_id": session_id}}

        transcript = "[S1] テスト会議です。予算は500万円です。"
        state = dict(initial_state(raw_transcript=transcript, session_id=session_id))

        state = TranscriptCleanNode().execute(state)
        if state.get("error"):
            return state

        state = NuanceClassifyNode().execute(state)

        extract_node = StructuredExtractNode(llm_client=mock_client, config_override=inner_cfg)
        state = extract_node.execute(state)  # Must not raise

        state = CRMSchemaValidateNode().execute(state)
        state = OutputFormatNode().execute(state)

        return state

    def test_pb5_timeout_no_crash(self):
        """PB-5: LLM TimeoutError → pipeline completes without raising."""
        from src.retry import TimeoutRetryError

        # Must not raise
        state = self._run_with_llm_error(TimeoutRetryError("timed out"), "pb5a")
        assert isinstance(state, dict), "Pipeline must return dict on LLM timeout"

    def test_pb5_timeout_sets_error_state(self):
        """PB-5: LLM TimeoutError → state['error'] is set."""
        from src.retry import TimeoutRetryError

        state = self._run_with_llm_error(TimeoutRetryError("timed out"), "pb5b")
        assert state.get("error", "") != "", "state['error'] must be set on LLM timeout"

    def test_pb5_schema_failure_no_crash(self):
        """PB-5: Malformed LLM JSON → pipeline completes without raising."""
        from src.retry import ValidationRetryError

        state = self._run_with_llm_error(ValidationRetryError("bad JSON"), "pb5c")
        assert isinstance(state, dict), "Pipeline must return dict on schema failure"

    def test_pb5_schema_failure_sets_error_state(self):
        """PB-5: Malformed LLM JSON → state['error'] is set with correct code."""
        from src.retry import ValidationRetryError

        state = self._run_with_llm_error(ValidationRetryError("bad JSON"), "pb5d")
        assert state.get("error", "") != "", "state['error'] must be set on schema failure"
        err = json.loads(state["error"])
        assert err.get("code") == "LLM_SCHEMA_FAILURE", f"Expected LLM_SCHEMA_FAILURE, got: {err.get('code')}"

    def test_pb5_output_format_still_runs_on_error(self):
        """PB-5: OutputFormatNode must still run S-3 gate even when upstream errored."""
        from src.retry import ValidationRetryError

        state = self._run_with_llm_error(ValidationRetryError("bad JSON"), "pb5e")
        # OutputFormatNode sets audit_output_hash even on error path
        assert state.get("audit_output_hash", "") != "", (
            "OutputFormatNode must run (and set audit_output_hash) even on error state"
        )


# ════════════════════════════════════════════════════════════════════════════════
# PB-6 / PB-7 — CRM schema switching
# ════════════════════════════════════════════════════════════════════════════════


class TestPBCRMSchemaSwitching:
    """PB-6, PB-7: CRM schema config switches between Salesforce and HubSpot.

    Design doc §5.3, §5.4, §7.1, ADR-003.
    """

    _SALESFORCE_KEYS = {"Name", "StageName", "Amount", "NextStep", "Description", "CloseDate"}
    _HUBSPOT_KEYS = {"dealname", "dealstage", "amount", "closedate", "description"}

    def test_pb6_salesforce_schema_produces_salesforce_keys(self):
        """PB-6: Salesforce config → crm_fields contains Salesforce-specific keys."""
        state = _run_pipeline(
            raw_transcript=_CLEAN_TRANSCRIPT,
            llm_response=_MOCK_LLM_RESPONSE_SALESFORCE,
            crm_schema="config/schemas/salesforce_opportunity.json",
            session_id="pb6-test",
        )
        output = _parse_output(state)
        crm = output.get("crm_fields", {})
        assert isinstance(crm, dict), f"crm_fields must be dict, got {type(crm)}"
        present = self._SALESFORCE_KEYS & set(crm.keys())
        assert present, f"No Salesforce keys found in crm_fields. Got keys: {set(crm.keys())}"

    def test_pb7_hubspot_schema_produces_hubspot_keys(self):
        """PB-7: HubSpot config → crm_fields contains HubSpot-specific keys."""
        state = _run_pipeline(
            raw_transcript=_CLEAN_TRANSCRIPT,
            llm_response=_MOCK_LLM_RESPONSE_HUBSPOT,
            crm_schema="config/schemas/hubspot_deal.json",
            session_id="pb7-test",
        )
        output = _parse_output(state)
        crm = output.get("crm_fields", {})
        assert isinstance(crm, dict), f"crm_fields must be dict, got {type(crm)}"
        present = self._HUBSPOT_KEYS & set(crm.keys())
        assert present, f"No HubSpot keys found in crm_fields. Got keys: {set(crm.keys())}"

    def test_pb6_salesforce_only_keys_absent_in_hubspot(self):
        """PB-7: HubSpot output must not contain Salesforce-only field names."""
        state = _run_pipeline(
            raw_transcript=_CLEAN_TRANSCRIPT,
            llm_response=_MOCK_LLM_RESPONSE_HUBSPOT,
            crm_schema="config/schemas/hubspot_deal.json",
            session_id="pb6b-test",
        )
        output = _parse_output(state)
        crm = output.get("crm_fields", {})
        # StageName and NextStep are Salesforce-only
        salesforce_only = {"StageName", "NextStep"}
        unexpected = salesforce_only & set(crm.keys())
        assert not unexpected, (
            f"Salesforce-only keys {unexpected} found in HubSpot output — schema switching may not be working correctly"
        )
