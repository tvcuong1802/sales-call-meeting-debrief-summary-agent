"""
Tests for src/nodes/output_format.py — OutputFormatNode.

Design doc §5.5.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.nodes.output_format import OutputFormatNode
from src.schemas.state import initial_state
from framework.errors import SecurityViolationError


def _make_state(**overrides) -> dict:
    state = initial_state("test transcript", "sess-test")
    state.update(overrides)
    return state


class TestOutputFormatNode:
    """Tests for OutputFormatNode."""

    def setup_method(self):
        self.node = OutputFormatNode()

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_happy_path_output_json_set(self):
        """On valid state, output_json is set to JSON string."""
        state = _make_state(
            summary="Meeting went well.",
            crm_fields=json.dumps({"Name": "Acme", "Amount": 50000.0}),
            email_draft="Dear [S1], thank you for today's meeting.",
            crm_validated=True,
            pii_masked=False,
            invocation_id="abc123",
        )
        result = self.node.execute(state)
        assert result["output_json"] != ""
        parsed = json.loads(result["output_json"])
        assert "summary" in parsed
        assert "crm_fields" in parsed
        assert "email_draft" in parsed
        assert "meta" in parsed

    def test_audit_output_hash_set(self):
        """audit_output_hash is SHA-256 of output_json."""
        import hashlib

        state = _make_state(
            summary="Summary here.",
            email_draft="Dear customer,",
        )
        result = self.node.execute(state)
        expected = hashlib.sha256(result["output_json"].encode("utf-8")).hexdigest()
        assert result["audit_output_hash"] == expected

    def test_meta_fields_included(self):
        """meta block contains crm_validated, pii_masked, invocation_id."""
        state = _make_state(
            crm_validated=True,
            pii_masked=True,
            invocation_id="inv-001",
        )
        result = self.node.execute(state)
        parsed = json.loads(result["output_json"])
        assert parsed["meta"]["crm_validated"] is True
        assert parsed["meta"]["pii_masked"] is True
        assert parsed["meta"]["invocation_id"] == "inv-001"

    # ── S-3 gate — runs on error ──────────────────────────────────────────────

    def test_s3_runs_even_when_error_set(self):
        """S-3 gate and output formatting run even when state['error'] is non-empty."""
        state = _make_state(
            error=json.dumps({"code": "LLM_TIMEOUT", "attempt": 2}),
            summary="",
            email_draft="",
        )
        result = self.node.execute(state)
        # output_json must still be set
        assert result["output_json"] != ""
        parsed = json.loads(result["output_json"])
        assert "error" in parsed
        assert parsed["error"]["code"] == "LLM_TIMEOUT"

    # ── S-3 credential detection ──────────────────────────────────────────────

    def test_bearer_token_in_summary_raises(self):
        """Bearer token in summary raises SecurityViolationError."""
        state = _make_state(
            summary="Use this token: Bearer abcdefghijklmnopqrstuvwxyz123456789",
        )
        with pytest.raises(SecurityViolationError):
            self.node.execute(state)

    def test_api_key_in_email_raises(self):
        """API key pattern in email_draft raises SecurityViolationError."""
        state = _make_state(
            email_draft="Here is the key: sk-abcdefghijklmnopqrstuvwxyz12345",
        )
        with pytest.raises(SecurityViolationError):
            self.node.execute(state)

    def test_clean_output_no_security_violation(self):
        """Clean output without credentials passes S-3 gate."""
        state = _make_state(
            summary="We discussed budget and timeline.",
            email_draft="Dear Customer, thank you for your time.",
        )
        result = self.node.execute(state)
        assert result["error"] == ""

    # ── Redact patterns ───────────────────────────────────────────────────────

    def test_redact_patterns_applied(self):
        """Configured redact_patterns replace matching content with [REDACTED]."""
        state = _make_state(
            summary="Project Falcon is moving forward.",
            email_draft="Regarding Project Falcon timeline...",
        )
        config = {"security": {"redact_patterns": [r"Project Falcon"]}}
        result = OutputFormatNode(config=config).execute(state)
        parsed = json.loads(result["output_json"])
        assert "Project Falcon" not in parsed["summary"]
        assert "[REDACTED]" in parsed["summary"]

    # ── crm_fields deserialization ────────────────────────────────────────────

    def test_crm_fields_deserialized_in_output(self):
        """crm_fields in output_json is a dict (not a JSON string)."""
        state = _make_state(crm_fields=json.dumps({"Name": "Acme", "Amount": 10000.0}))
        result = self.node.execute(state)
        parsed = json.loads(result["output_json"])
        assert isinstance(parsed["crm_fields"], dict)
        assert parsed["crm_fields"]["Name"] == "Acme"

    def test_audit_hash_is_64_char_hex(self):
        """audit_output_hash is 64-character hex string (SHA-256)."""
        state = _make_state(summary="Summary.", email_draft="Email.")
        result = self.node.execute(state)
        assert len(result["audit_output_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in result["audit_output_hash"])
