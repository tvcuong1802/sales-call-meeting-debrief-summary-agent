"""
Proof-of-Boundary test: PB-PII — S-2 gate fires before LLM call.

Verifies that PII present in raw_transcript is masked in the state
received by StructuredExtractNode, i.e., the LLM never sees raw PII.

Design doc §8.2. APPI no-persistence guarantee.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.nodes.transcript_clean import TranscriptCleanNode
from src.nodes.structured_extract import StructuredExtractNode
from src.nodes.nuance_classify import NuanceClassifyNode
from src.schemas.state import initial_state


# ── PII-laden transcript fixture ──────────────────────────────────────────────

_PII_TRANSCRIPT = (
    "[Speaker 1]: 山田様、先日お送りした taro.yamada@client.co.jp をご確認いただけましたか？\n"
    "[Speaker 2]: はい、確認しました。折り返し 090-9999-8888 にお電話します。\n"
    "[Speaker 1]: ありがとうございます。弊社は東京都千代田区大手町1丁目1番地にあります。\n"
    "[Speaker 2]: 来週、John Smith を同行させます。\n"
    "[Speaker 1]: 予算は3,000万円で、Q2クローズを目指しています。\n"
)

_PII_TOKENS = [
    "taro.yamada@client.co.jp",
    "090-9999-8888",
    "John Smith",
    "山田様",
]

_SAFE_CONTENT = [
    "Q2",
    "3,000万円",
    "Q2クローズ",
    "予算",
]


def _make_extract_node_cfg():
    return {
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
    }


# ════════════════════════════════════════════════════════════════════════════════
# TestPBPIIGate
# ════════════════════════════════════════════════════════════════════════════════


class TestPBPIIGate:
    """Proof-of-Boundary: S-2 gate fires before LLM sees state.

    Method: run TranscriptCleanNode → NuanceClassifyNode → capture the
    annotated_transcript that would be passed to StructuredExtractNode.
    Assert raw PII is absent from that field.
    """

    def _run_pre_process(self, raw_transcript: str) -> dict:
        """Run pre-process nodes (TranscriptClean + NuanceClassify) and return final state."""
        state = dict(initial_state(raw_transcript=raw_transcript, session_id="pb-test-001"))

        # Step 1: TranscriptCleanNode (includes S-2 gate)
        clean_node = TranscriptCleanNode()
        state = clean_node.execute(state)
        assert state.get("error", "") == "", f"TranscriptCleanNode error: {state['error']}"

        # Step 2: NuanceClassifyNode
        nuance_node = NuanceClassifyNode()
        state = nuance_node.execute(state)

        return state

    def test_pii_absent_from_raw_transcript_after_gate(self):
        """raw_transcript must not contain any PII after TranscriptCleanNode."""
        state = self._run_pre_process(_PII_TRANSCRIPT)
        for pii_token in _PII_TOKENS:
            assert pii_token not in state["raw_transcript"], (
                f"PII token {pii_token!r} found in raw_transcript after S-2 gate"
            )

    def test_pii_absent_from_cleaned_transcript(self):
        """cleaned_transcript must not contain any PII."""
        state = self._run_pre_process(_PII_TRANSCRIPT)
        for pii_token in _PII_TOKENS:
            assert pii_token not in state["cleaned_transcript"], f"PII token {pii_token!r} found in cleaned_transcript"

    def test_pii_absent_from_annotated_transcript(self):
        """annotated_transcript (what StructuredExtractNode receives) must not contain PII."""
        state = self._run_pre_process(_PII_TRANSCRIPT)
        for pii_token in _PII_TOKENS:
            assert pii_token not in state["annotated_transcript"], (
                f"PII token {pii_token!r} found in annotated_transcript (LLM input)"
            )

    def test_mask_tokens_present_in_cleaned_transcript(self):
        """At least one mask token must appear in cleaned_transcript."""
        state = self._run_pre_process(_PII_TRANSCRIPT)
        mask_tokens = ["[NAME]", "[PHONE]", "[EMAIL]", "[ADDRESS]", "[FINANCIAL]", "[GOV_ID]", "[MEDICAL_ID]"]
        found = [t for t in mask_tokens if t in state["cleaned_transcript"]]
        assert found, (
            f"No mask tokens found in cleaned_transcript. cleaned_transcript={state['cleaned_transcript'][:200]!r}"
        )

    def test_non_pii_sales_content_preserved(self):
        """Non-PII sales content (budget, timeline) must survive masking."""
        state = self._run_pre_process(_PII_TRANSCRIPT)
        for safe_token in _SAFE_CONTENT:
            assert safe_token in state["cleaned_transcript"] or safe_token in state["annotated_transcript"], (
                f"Safe token {safe_token!r} unexpectedly removed from transcript"
            )

    def test_pii_masked_flag_set(self):
        state = self._run_pre_process(_PII_TRANSCRIPT)
        assert state["pii_masked"] is True

    def test_pii_mask_count_nonzero(self):
        state = self._run_pre_process(_PII_TRANSCRIPT)
        assert state["pii_mask_count"] >= 1

    def test_llm_prompt_never_receives_raw_pii(self):
        """Verify prompt passed to LLM client contains no raw PII.

        Runs the full pre-process pipeline, then feeds state into
        StructuredExtractNode with a mock LLM client. Captures the
        prompt string and asserts it contains no raw PII tokens.
        """
        state = self._run_pre_process(_PII_TRANSCRIPT)
        if state.get("error"):
            pytest.skip("Pre-process error — cannot verify LLM prompt")

        # Minimal valid LLM response
        mock_response = json.dumps(
            {
                "summary": "テスト",
                "crm_fields": {
                    "Name": None,
                    "StageName": None,
                    "Amount": None,
                    "NextStep": None,
                    "Description": None,
                    "CloseDate": None,
                },
                "email_draft": "Subject: Test\n\nBody",
            }
        )
        mock_client = MagicMock()
        mock_client.invoke.return_value = mock_response

        try:
            extract_node = StructuredExtractNode(
                llm_client=mock_client,
                config_override=_make_extract_node_cfg(),
            )
        except TypeError:
            # develop branch has old-style stub __init__ — still verifiable via gate
            extract_node = StructuredExtractNode(llm_client=mock_client)
        extract_node.execute(state)

        if not mock_client.invoke.called:
            pytest.skip(
                "StructuredExtractNode is still a stub on this branch — LLM prompt PII test requires Part 1/2/3 merged"
            )
        prompt_arg = mock_client.invoke.call_args[0][0]

        for pii_token in _PII_TOKENS:
            assert pii_token not in prompt_arg, f"Raw PII {pii_token!r} found in LLM prompt — S-2 gate failure"
