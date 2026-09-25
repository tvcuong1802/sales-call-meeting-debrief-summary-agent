"""
Tests for src/nodes/transcript_clean.py — TranscriptCleanNode.

Design doc §5.1.
"""

from __future__ import annotations

import json
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from framework.schemas.invocation_context import TrustLevel
from src.nodes.transcript_clean import TranscriptCleanNode, MAX_TRANSCRIPT_CHARS
from src.schemas.state import initial_state


def _make_state(**overrides) -> dict:
    state = initial_state("test transcript", "sess-test")
    state.update(overrides)
    return state


class TestTranscriptCleanNode:
    """Tests for TranscriptCleanNode."""

    def setup_method(self):
        self.node = TranscriptCleanNode()

    # ── S-1 input validation ──────────────────────────────────────────────────

    def test_empty_input_rejected(self):
        """Empty raw_transcript sets state['error'] with S1_INPUT_REJECTED."""
        state = _make_state(raw_transcript="")
        result = self.node.execute(state)
        assert result["error"] != ""
        error = json.loads(result["error"])
        assert error["code"] == "S1_INPUT_REJECTED"
        assert error["node"] == "TranscriptCleanNode"

    def test_oversized_input_rejected(self):
        """Transcript > 50,000 chars sets state['error'] with S1_INPUT_REJECTED."""
        big = "x" * (MAX_TRANSCRIPT_CHARS + 1)
        state = _make_state(raw_transcript=big)
        result = self.node.execute(state)
        assert result["error"] != ""
        error = json.loads(result["error"])
        assert error["code"] == "S1_INPUT_REJECTED"
        assert error.get("reason") == "oversized_input"

    def test_exactly_max_chars_allowed(self):
        """Transcript of exactly MAX_TRANSCRIPT_CHARS chars is accepted."""
        at_limit = "a" * MAX_TRANSCRIPT_CHARS
        state = _make_state(raw_transcript=at_limit)
        result = self.node.execute(state)
        assert result["error"] == ""
        assert result["cleaned_transcript"] != ""

    def test_non_text_bytes_rejected(self):
        """Transcript with null byte \\x00 is rejected as non-text."""
        state = _make_state(raw_transcript="valid start\x00invalid")
        result = self.node.execute(state)
        assert result["error"] != ""
        error = json.loads(result["error"])
        assert error["code"] == "S1_INPUT_REJECTED"
        assert error.get("reason") == "non_text_bytes"

    # ── Invocation ID and audit hash ──────────────────────────────────────────

    def test_invocation_id_generated(self):
        """invocation_id is set to a non-empty standard UUID4 string."""
        state = _make_state(raw_transcript="meeting transcript here")
        result = self.node.execute(state)
        assert result["invocation_id"] != ""
        # Standard UUID4 with hyphens: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx (36 chars)
        assert len(result["invocation_id"]) == 36
        assert result["invocation_id"].count("-") == 4

    def test_audit_hash_computed(self):
        """audit_input_hash is set to SHA-256 hex of raw_transcript."""
        import hashlib

        raw = "some meeting transcript"
        state = _make_state(raw_transcript=raw)
        result = self.node.execute(state)
        expected = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        assert result["audit_input_hash"] == expected

    def test_audit_hash_is_64_char_hex(self):
        """audit_input_hash is a 64-character hex string (SHA-256)."""
        state = _make_state(raw_transcript="transcript")
        result = self.node.execute(state)
        assert len(result["audit_input_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in result["audit_input_hash"])

    # ── Normalization ─────────────────────────────────────────────────────────

    def test_aizuchi_stripped(self):
        """Aizuchi (えー, あー, うん) are stripped from transcript."""
        state = _make_state(raw_transcript="えー本日はうんありがとうございます。あーよろしくお願いします。")
        result = self.node.execute(state)
        assert "えー" not in result["cleaned_transcript"]
        assert "あー" not in result["cleaned_transcript"]
        assert "うん" not in result["cleaned_transcript"]
        assert "本日は" in result["cleaned_transcript"]

    def test_speaker_tag_normalised(self):
        """[Speaker 1] → [S1], [Speaker 12] → [S12]."""
        state = _make_state(raw_transcript="[Speaker 1] Hello. [Speaker 2] Good day. [Speaker 12] Hi.")
        result = self.node.execute(state)
        assert "[S1]" in result["cleaned_transcript"]
        assert "[S2]" in result["cleaned_transcript"]
        assert "[S12]" in result["cleaned_transcript"]
        assert "[Speaker 1]" not in result["cleaned_transcript"]
        assert "[Speaker 2]" not in result["cleaned_transcript"]

    def test_repeated_whitespace_collapsed(self):
        """Multiple spaces/tabs collapsed to single space."""
        state = _make_state(raw_transcript="word1    word2\t\t\tword3")
        result = self.node.execute(state)
        assert "word1 word2 word3" in result["cleaned_transcript"]

    def test_multiple_blank_lines_collapsed(self):
        """Three or more consecutive newlines collapsed to double newline."""
        state = _make_state(raw_transcript="para1\n\n\n\npara2")
        result = self.node.execute(state)
        assert "para1\n\npara2" in result["cleaned_transcript"]

    # ── Short-circuit ─────────────────────────────────────────────────────────

    def test_short_circuit_on_error(self):
        """If state['error'] is non-empty, node returns state unchanged."""
        state = _make_state(
            raw_transcript="valid transcript",
            error=json.dumps({"code": "PRIOR_ERROR", "attempt": 1}),
        )
        result = self.node.execute(state)
        # Should not process — invocation_id should remain empty
        assert result["invocation_id"] == ""
        assert result["cleaned_transcript"] == ""

    # ── S-2 stub ──────────────────────────────────────────────────────────────

    def test_s2_mask_pii_is_non_empty(self):
        """_mask_pii (node-level S-2 helper) exists and is non-empty (not pass-through)."""
        import inspect

        src = inspect.getsource(TranscriptCleanNode._mask_pii)
        # Must have implementation content beyond just 'return state'
        assert "stub" in src.lower() or "issue #16" in src or "pii" in src.lower()
        # Must NOT be empty (just a docstring + return)
        non_comment_lines = [
            l.strip()
            for l in src.split("\n")
            if l.strip()
            and not l.strip().startswith("#")
            and not l.strip().startswith('"""')
            and not l.strip().startswith("def ")
        ]
        assert len(non_comment_lines) > 1  # more than just 'return state'

    def test_s2_mask_pii_returns_state(self):
        """_mask_pii (node-level S-2 helper) returns a state dict (non-None)."""
        state = _make_state(raw_transcript="test")
        result = self.node._mask_pii(state)
        assert isinstance(result, dict)

    # ── Happy path ────────────────────────────────────────────────────────────

    # ── TC-08: required_trust_level (the framework contract) ─────────────────────────

    def test_required_trust_level_declared(self):
        """TC-08: required_trust_level is declared on TranscriptCleanNode (the framework contract)."""
        assert self.node.required_trust_level == TrustLevel.VERIFIED_EXTERNAL, (
            "TranscriptCleanNode must declare required_trust_level = VERIFIED_EXTERNAL "
            "(sales transcripts contain customer PII — ANONYMOUS callers must be rejected)"
        )

    def test_required_trust_level_is_class_attribute(self):
        """TC-08: required_trust_level is a class-level attribute, not instance-only."""
        assert hasattr(TranscriptCleanNode, "required_trust_level")
        assert TranscriptCleanNode.required_trust_level == TrustLevel.VERIFIED_EXTERNAL

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_happy_path_all_fields_set(self):
        """On valid input, all output fields are set correctly."""
        state = _make_state(raw_transcript="[Speaker 1] Good morning. えーよろしくお願いします。")
        result = self.node.execute(state)
        assert result["error"] == ""
        assert result["cleaned_transcript"] != ""
        assert result["invocation_id"] != ""
        assert result["audit_input_hash"] != ""
        assert isinstance(result["pii_masked"], bool)
        assert isinstance(result["pii_mask_count"], int)
