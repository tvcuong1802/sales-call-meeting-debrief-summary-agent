"""
Unit tests for S-2 PII detection gate — CMN-C1-042.

Tests _security_gate_input() in TranscriptCleanNode.
Verifies APPI compliance: all PII categories masked before LLM call,
no raw PII in trace events, correct state fields set.

Design doc §8.2, the security rules
"""

from __future__ import annotations

import json
from unittest.mock import patch


from src.nodes.transcript_clean import (
    TranscriptCleanNode,
)
from src.schemas.state import initial_state


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_node() -> TranscriptCleanNode:
    return TranscriptCleanNode()


def _make_state(raw_transcript: str = "", **kwargs) -> dict:
    s = dict(initial_state(raw_transcript=raw_transcript or "test", session_id="sess-001"))
    s["raw_transcript"] = raw_transcript
    s.update(kwargs)
    return s


def _gate(raw: str) -> dict:
    """Run only the S-2 gate on a raw transcript string. Returns result state."""
    node = _make_node()
    state = _make_state(raw_transcript=raw)
    return node._mask_pii(state)


# ── PII fixture transcripts ────────────────────────────────────────────────────

PHONE_JP_MOBILE = "営業の田中です。090-1234-5678 にお電話ください。"
PHONE_JP_LANDLINE = "本社は 03-9876-5432 です。"
PHONE_INTL = "海外からは +81-90-1234-5678 へ。"
PHONE_BARE = "番号は 09012345678 です。"

EMAIL_SIMPLE = "ご連絡は taro.tanaka@example.co.jp まで。"
EMAIL_PLUS = "担当者は sales+info@corp.example.com です。"

GOV_ID_MY_NUMBER = "マイナンバーは 123456789012 です。"
GOV_ID_SSN = "SSN: 123-45-6789 を提出してください。"

FINANCIAL_CARD_4X4 = "カード番号は 4111-1111-1111-1111 です。"
FINANCIAL_CARD_BARE = "番号 4111111111111111 を入力してください。"
FINANCIAL_JP_BANK = "口座番号は 123-4567890 です。"

ADDRESS_JP = "本社所在地：東京都渋谷区代々木1丁目2番3号"
ADDRESS_EN = "Our office is at 123 Main Street, downtown."
ADDRESS_EN_TYPE2 = "Visit us at 45 Oak Avenue for the demo."

NAME_JP_SAMA = "田中様がご来社されました。"
NAME_JP_SAN = "鈴木さんから連絡がありました。"
NAME_JP_SHI = "山田氏の提案を検討します。"
NAME_EN_PAIR = "John Smith will join the call."
NAME_EN_THREE = "Mary Jane Watson confirmed the meeting."

MEDICAL_KARTE = "カルテ番号 12345678 の患者様です。"
MEDICAL_MR = "Patient MR-00123456 admitted."

# Clean transcript — no PII (should pass through unmodified)
CLEAN_TRANSCRIPT = (
    "[Speaker 1]: 本日はよろしくお願いします。\n"
    "[Speaker 2]: こちらこそよろしくお願いします。\n"
    "[Speaker 1]: 御社の在庫管理ソリューションについて伺いたいのですが。\n"
    "[Speaker 2]: はい、弊社のソリューションは月次コスト30%削減実績があります。\n"
    "[Speaker 1]: 導入スケジュールはどのくらいかかりますか？\n"
    "[Speaker 2]: 標準で3ヶ月、最短で6週間での導入が可能です。\n"
)

# Company name that should NOT be masked (田中商事 has no honorific suffix)
COMPANY_NAME_SAFE = "田中商事株式会社は本件のスポンサーです。"

# Meeting room / floor references that should NOT trigger address masking
MEETING_ROOM_SAFE = "会議は3階の会議室Aで行います。Room 301 is booked."


# ════════════════════════════════════════════════════════════════════════════════
# TestPhoneDetection
# ════════════════════════════════════════════════════════════════════════════════


class TestPhoneDetection:
    def test_jp_mobile_masked(self):
        result = _gate(PHONE_JP_MOBILE)
        assert "090-1234-5678" not in result["raw_transcript"]
        assert "[PHONE]" in result["raw_transcript"]

    def test_jp_landline_masked(self):
        result = _gate(PHONE_JP_LANDLINE)
        assert "03-9876-5432" not in result["raw_transcript"]
        assert "[PHONE]" in result["raw_transcript"]

    def test_international_masked(self):
        result = _gate(PHONE_INTL)
        assert "+81-90-1234-5678" not in result["raw_transcript"]
        assert "[PHONE]" in result["raw_transcript"]

    def test_bare_11digit_masked(self):
        result = _gate(PHONE_BARE)
        assert "09012345678" not in result["raw_transcript"]
        assert "[PHONE]" in result["raw_transcript"]

    def test_pii_masked_flag_set(self):
        assert _gate(PHONE_JP_MOBILE)["pii_masked"] is True

    def test_pii_mask_count_incremented(self):
        assert _gate(PHONE_JP_MOBILE)["pii_mask_count"] >= 1


# ════════════════════════════════════════════════════════════════════════════════
# TestEmailDetection
# ════════════════════════════════════════════════════════════════════════════════


class TestEmailDetection:
    def test_simple_email_masked(self):
        result = _gate(EMAIL_SIMPLE)
        assert "taro.tanaka@example.co.jp" not in result["raw_transcript"]
        assert "[EMAIL]" in result["raw_transcript"]

    def test_plus_address_masked(self):
        result = _gate(EMAIL_PLUS)
        assert "sales+info@corp.example.com" not in result["raw_transcript"]
        assert "[EMAIL]" in result["raw_transcript"]

    def test_pii_masked_flag_set(self):
        assert _gate(EMAIL_SIMPLE)["pii_masked"] is True


# ════════════════════════════════════════════════════════════════════════════════
# TestGovernmentIDDetection
# ════════════════════════════════════════════════════════════════════════════════


class TestGovernmentIDDetection:
    def test_my_number_12digit_masked(self):
        result = _gate(GOV_ID_MY_NUMBER)
        assert "123456789012" not in result["raw_transcript"]
        assert "[GOV_ID]" in result["raw_transcript"]

    def test_ssn_masked(self):
        result = _gate(GOV_ID_SSN)
        assert "123-45-6789" not in result["raw_transcript"]
        assert "[GOV_ID]" in result["raw_transcript"]

    def test_pii_masked_flag_set(self):
        assert _gate(GOV_ID_MY_NUMBER)["pii_masked"] is True


# ════════════════════════════════════════════════════════════════════════════════
# TestFinancialIDDetection
# ════════════════════════════════════════════════════════════════════════════════


class TestFinancialIDDetection:
    def test_credit_card_4x4_masked(self):
        result = _gate(FINANCIAL_CARD_4X4)
        assert "4111-1111-1111-1111" not in result["raw_transcript"]
        assert "[FINANCIAL]" in result["raw_transcript"]

    def test_credit_card_bare_16digit_masked(self):
        result = _gate(FINANCIAL_CARD_BARE)
        assert "4111111111111111" not in result["raw_transcript"]
        assert "[FINANCIAL]" in result["raw_transcript"]

    def test_jp_bank_account_masked(self):
        result = _gate(FINANCIAL_JP_BANK)
        assert "123-4567890" not in result["raw_transcript"]
        assert "[FINANCIAL]" in result["raw_transcript"]

    def test_pii_masked_flag_set(self):
        assert _gate(FINANCIAL_CARD_4X4)["pii_masked"] is True


# ════════════════════════════════════════════════════════════════════════════════
# TestAddressDetection
# ════════════════════════════════════════════════════════════════════════════════


class TestAddressDetection:
    def test_jp_address_masked(self):
        result = _gate(ADDRESS_JP)
        assert "渋谷区代々木1丁目2番3号" not in result["raw_transcript"]
        assert "[ADDRESS]" in result["raw_transcript"]

    def test_en_address_street_masked(self):
        result = _gate(ADDRESS_EN)
        assert "123 Main Street" not in result["raw_transcript"]
        assert "[ADDRESS]" in result["raw_transcript"]

    def test_en_address_avenue_masked(self):
        result = _gate(ADDRESS_EN_TYPE2)
        assert "45 Oak Avenue" not in result["raw_transcript"]
        assert "[ADDRESS]" in result["raw_transcript"]

    def test_meeting_room_not_masked(self):
        """Floor/room references must not trigger address masking."""
        result = _gate(MEETING_ROOM_SAFE)
        assert "[ADDRESS]" not in result["raw_transcript"]
        assert result["pii_masked"] is False

    def test_pii_masked_flag_set(self):
        assert _gate(ADDRESS_JP)["pii_masked"] is True


# ════════════════════════════════════════════════════════════════════════════════
# TestNameDetection
# ════════════════════════════════════════════════════════════════════════════════


class TestNameDetection:
    def test_jp_name_sama_masked(self):
        result = _gate(NAME_JP_SAMA)
        assert "田中様" not in result["raw_transcript"]
        assert "[NAME]" in result["raw_transcript"]

    def test_jp_name_san_masked(self):
        result = _gate(NAME_JP_SAN)
        assert "鈴木さん" not in result["raw_transcript"]
        assert "[NAME]" in result["raw_transcript"]

    def test_jp_name_shi_masked(self):
        result = _gate(NAME_JP_SHI)
        assert "山田氏" not in result["raw_transcript"]
        assert "[NAME]" in result["raw_transcript"]

    def test_en_name_pair_masked(self):
        result = _gate(NAME_EN_PAIR)
        assert "John Smith" not in result["raw_transcript"]
        assert "[NAME]" in result["raw_transcript"]

    def test_company_name_no_honorific_not_masked(self):
        """田中商事 (company name without honorific) must NOT be masked."""
        result = _gate(COMPANY_NAME_SAFE)
        assert "田中商事株式会社" in result["raw_transcript"]
        assert result["pii_masked"] is False

    def test_pii_masked_flag_set(self):
        assert _gate(NAME_JP_SAMA)["pii_masked"] is True


# ════════════════════════════════════════════════════════════════════════════════
# TestMedicalIDDetection
# ════════════════════════════════════════════════════════════════════════════════


class TestMedicalIDDetection:
    def test_karte_number_masked(self):
        result = _gate(MEDICAL_KARTE)
        assert "12345678" not in result["raw_transcript"]
        assert "[MEDICAL_ID]" in result["raw_transcript"]

    def test_mr_number_masked(self):
        result = _gate(MEDICAL_MR)
        assert "MR-00123456" not in result["raw_transcript"]
        assert "[MEDICAL_ID]" in result["raw_transcript"]

    def test_pii_masked_flag_set(self):
        assert _gate(MEDICAL_KARTE)["pii_masked"] is True


# ════════════════════════════════════════════════════════════════════════════════
# TestCleanTranscript (no false positives)
# ════════════════════════════════════════════════════════════════════════════════


class TestCleanTranscript:
    def test_clean_transcript_passes_through_unmodified(self):
        """A normal sales transcript with no PII must not be altered."""
        result = _gate(CLEAN_TRANSCRIPT)
        assert result["raw_transcript"] == CLEAN_TRANSCRIPT

    def test_pii_masked_false_on_clean(self):
        assert _gate(CLEAN_TRANSCRIPT)["pii_masked"] is False

    def test_pii_mask_count_zero_on_clean(self):
        assert _gate(CLEAN_TRANSCRIPT)["pii_mask_count"] == 0


# ════════════════════════════════════════════════════════════════════════════════
# TestMultiplePIITypes
# ════════════════════════════════════════════════════════════════════════════════


class TestMultiplePIITypes:
    _MULTI_PII = (
        "田中様、先ほどのメールアドレス taro@example.com と "
        "電話番号 090-1111-2222 を確認しました。"
        "マイナンバーは 123456789012 です。"
    )

    def test_all_types_masked(self):
        result = _gate(self._MULTI_PII)
        assert "taro@example.com" not in result["raw_transcript"]
        assert "090-1111-2222" not in result["raw_transcript"]
        assert "123456789012" not in result["raw_transcript"]
        assert "田中様" not in result["raw_transcript"]

    def test_mask_count_reflects_all_tokens(self):
        result = _gate(self._MULTI_PII)
        # At least 4 tokens: [NAME] + [EMAIL] + [PHONE] + [GOV_ID]
        assert result["pii_mask_count"] >= 4

    def test_pii_masked_true(self):
        assert _gate(self._MULTI_PII)["pii_masked"] is True


# ════════════════════════════════════════════════════════════════════════════════
# TestS4TraceEvent — APPI no-persistence
# ════════════════════════════════════════════════════════════════════════════════


class TestS4TraceEvent:
    def test_pii_detected_event_emitted_when_pii_present(self):
        node = _make_node()
        state = _make_state(raw_transcript=PHONE_JP_MOBILE)
        with patch(type(node).__module__ + ".emit_trace_event") as mock_emit:
            node._mask_pii(state)
        events = [c[0][0] for c in mock_emit.call_args_list]
        assert "pii_detected" in events

    def test_pii_detected_event_payload_has_required_fields(self):
        node = _make_node()
        state = _make_state(raw_transcript=PHONE_JP_MOBILE)
        with patch(type(node).__module__ + ".emit_trace_event") as mock_emit:
            node._mask_pii(state)
        pii_calls = [c for c in mock_emit.call_args_list if c[0][0] == "pii_detected"]
        assert len(pii_calls) == 1
        payload = pii_calls[0][0][1]
        assert "pattern_types" in payload
        assert "count" in payload
        assert "action" in payload
        assert payload["action"] == "masked"
        assert payload["count"] >= 1

    def test_no_raw_pii_in_trace_payload(self):
        """APPI: raw PII content must NOT appear in trace event payload."""
        node = _make_node()
        state = _make_state(raw_transcript=PHONE_JP_MOBILE)
        with patch(type(node).__module__ + ".emit_trace_event") as mock_emit:
            node._mask_pii(state)
        # Collect all call args as strings
        all_payload_str = str(mock_emit.call_args_list)
        assert "090-1234-5678" not in all_payload_str

    def test_no_event_emitted_when_no_pii(self):
        """No pii_detected event must be emitted when transcript is clean."""
        node = _make_node()
        state = _make_state(raw_transcript=CLEAN_TRANSCRIPT)
        with patch(type(node).__module__ + ".emit_trace_event") as mock_emit:
            node._mask_pii(state)
        events = [c[0][0] for c in mock_emit.call_args_list]
        assert "pii_detected" not in events

    def test_pattern_types_list_no_pii_content(self):
        """pattern_types in trace must be category labels only (not matched strings)."""
        node = _make_node()
        state = _make_state(raw_transcript=EMAIL_SIMPLE)
        with patch(type(node).__module__ + ".emit_trace_event") as mock_emit:
            node._mask_pii(state)
        pii_calls = [c for c in mock_emit.call_args_list if c[0][0] == "pii_detected"]
        payload = pii_calls[0][0][1]
        for cat in payload["pattern_types"]:
            # category labels are uppercase strings like "EMAIL", "PHONE"
            assert cat.isupper(), f"Expected uppercase category label, got: {cat!r}"
            assert "@" not in cat
            assert "." not in cat


# ════════════════════════════════════════════════════════════════════════════════
# TestStateFields
# ════════════════════════════════════════════════════════════════════════════════


class TestStateFields:
    def test_raw_transcript_updated_in_returned_state(self):
        """raw_transcript in returned state must contain mask tokens, not raw PII."""
        original_state = _make_state(raw_transcript=PHONE_JP_MOBILE)
        result = _make_node()._mask_pii(original_state)
        assert result["raw_transcript"] != PHONE_JP_MOBILE
        assert "[PHONE]" in result["raw_transcript"]

    def test_original_state_not_mutated(self):
        """Input state dict must not be mutated in place."""
        original_state = _make_state(raw_transcript=PHONE_JP_MOBILE)
        original_raw = original_state["raw_transcript"]
        _ = _make_node()._mask_pii(original_state)
        assert original_state["raw_transcript"] == original_raw

    def test_pii_masked_true_when_pii_found(self):
        result = _make_node()._mask_pii(_make_state(raw_transcript=EMAIL_SIMPLE))
        assert result["pii_masked"] is True

    def test_pii_masked_false_when_clean(self):
        result = _make_node()._mask_pii(_make_state(raw_transcript=CLEAN_TRANSCRIPT))
        assert result["pii_masked"] is False

    def test_pii_mask_count_correct_single(self):
        result = _make_node()._mask_pii(_make_state(raw_transcript=EMAIL_SIMPLE))
        assert result["pii_mask_count"] == 1

    def test_empty_transcript_returns_unchanged(self):
        state = _make_state(raw_transcript="")
        result = _make_node()._mask_pii(state)
        assert result["pii_masked"] is False
        assert result["pii_mask_count"] == 0


# ════════════════════════════════════════════════════════════════════════════════
# TestUpstreamErrorShortCircuit
# ════════════════════════════════════════════════════════════════════════════════


class TestUpstreamErrorShortCircuit:
    def test_upstream_error_bypasses_gate(self):
        """If state already has error set, execute() must short-circuit before S-2."""
        node = _make_node()
        state = _make_state(
            raw_transcript=PHONE_JP_MOBILE,
            error=json.dumps({"code": "S1_INPUT_REJECTED", "attempt": 1, "node": "Upstream"}),
        )
        with patch(type(node).__module__ + ".emit_trace_event") as mock_emit:
            result = node.execute(state)
        # LLM gate must not have fired — raw_transcript unchanged
        assert result["raw_transcript"] == PHONE_JP_MOBILE
        events = [c[0][0] for c in mock_emit.call_args_list]
        assert "pii_detected" not in events


# ════════════════════════════════════════════════════════════════════════════════
# TestExecuteIntegration — gate fires within execute()
# ════════════════════════════════════════════════════════════════════════════════


class TestExecuteIntegration:
    def test_execute_masks_pii_before_cleaning(self):
        """Full execute() path: PII masked in raw_transcript, cleaned_transcript has mask tokens."""
        node = _make_node()
        state = _make_state(raw_transcript=f"[Speaker 1]: {PHONE_JP_MOBILE}")
        result = node.execute(state)
        assert result["error"] == ""
        # raw_transcript in state should have mask token
        assert "090-1234-5678" not in result["raw_transcript"]
        # cleaned_transcript (normalized version) also has no raw PII
        assert "090-1234-5678" not in result["cleaned_transcript"]

    def test_execute_pii_masked_flag_propagates(self):
        node = _make_node()
        state = _make_state(raw_transcript=f"[Speaker 1]: {EMAIL_SIMPLE}")
        result = node.execute(state)
        assert result["pii_masked"] is True
        assert result["pii_mask_count"] >= 1

    def test_execute_clean_transcript_pii_masked_false(self):
        node = _make_node()
        state = _make_state(raw_transcript=CLEAN_TRANSCRIPT)
        result = node.execute(state)
        assert result["pii_masked"] is False
        assert result["pii_mask_count"] == 0
        assert result["error"] == ""
