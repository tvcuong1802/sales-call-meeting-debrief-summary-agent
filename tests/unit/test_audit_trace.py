"""
Unit tests for S-4 audit trace — CMN-C1-042.

Verifies that each domain node emits DOMAIN trace events (not framework
backbone lifecycle events node_start/node_complete/node_error/node_skip,
which BaseNode.__call__ owns) per CoE Criterion #2.

Tests correlation_id extraction via the get_correlation_id() module helper.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from src.nodes.base import get_correlation_id
from src.nodes.transcript_clean import TranscriptCleanNode
from src.nodes.nuance_classify import NuanceClassifyNode
from src.nodes.crm_schema_validate import CRMSchemaValidateNode
from src.nodes.output_format import OutputFormatNode
from src.nodes.structured_extract import StructuredExtractNode
from src.schemas.state import initial_state


# ── get_correlation_id() module-helper tests (promoted from the removed shim) ──


class TestGetCorrelationId:
    def test_none_config_returns_empty(self):
        assert get_correlation_id(None) == ""

    def test_session_id_fallback(self):
        config = {"configurable": {"session_id": "test-session-123"}}
        assert get_correlation_id(config) == "test-session-123"

    def test_invocation_context_takes_priority(self):
        ctx = MagicMock()
        ctx.correlation_id = "ctx-corr-456"
        config = {"configurable": {"invocation_context": ctx, "session_id": "session-789"}}
        assert get_correlation_id(config) == "ctx-corr-456"

    def test_empty_config_returns_empty(self):
        assert get_correlation_id({}) == ""


# ── Per-node trace event tests ────────────────────────────────────────────────
#
# Domain nodes MUST emit only DOMAIN events. The framework backbone
# (BaseNode.__call__) owns the lifecycle events below — a domain node that
# emits them would duplicate/corrupt the audit trail (CoE Criterion #2).
_BACKBONE = {"node_start", "node_complete", "node_error", "node_skip"}


def _base_state(**kwargs):
    s = dict(initial_state(raw_transcript="Test transcript.", session_id="sess-001"))
    s.update(kwargs)
    return s


def _emitted(node, state):
    """Emit-capture for an ALREADY-CONFIGURED node.

    It used to take a `config` and pass it to execute(). The framework passes none, so a
    node had to be configured at construction; the caller builds it that way now.
    """
    with patch(type(node).__module__ + ".emit_trace_event") as mock_emit:
        node.execute(state)
    return [c[0][0] for c in mock_emit.call_args_list]


def _assert_domain_only(events):
    """At least one domain event, and zero backbone lifecycle events."""
    assert events, "node emitted no trace events at all"
    assert not (_BACKBONE & set(events)), f"node emitted forbidden backbone event(s): {_BACKBONE & set(events)}"


class TestTranscriptCleanNodeTrace:
    def test_emits_domain_event_on_success(self):
        events = _emitted(TranscriptCleanNode(), _base_state())
        _assert_domain_only(events)
        assert "transcript_cleaned" in events

    def test_emits_domain_error_on_empty_input(self):
        events = _emitted(TranscriptCleanNode(), _base_state(raw_transcript=""))
        _assert_domain_only(events)
        assert "transcript_clean_error" in events


class TestNuanceClassifyNodeTrace:
    def test_emits_domain_event_on_success(self):
        state = _base_state(cleaned_transcript="テスト", invocation_id="inv-001")
        events = _emitted(NuanceClassifyNode(), state)
        _assert_domain_only(events)
        assert "nuance_classified" in events


class TestCRMSchemaValidateNodeTrace:
    def test_emits_domain_event_on_success(self):
        node = CRMSchemaValidateNode()
        state = _base_state(crm_fields="{}", invocation_id="inv-002")
        with patch(type(node).__module__ + ".emit_trace_event") as mock_emit:
            with patch.object(
                node, "_load_schema", return_value={"type": "object", "properties": {}, "additionalProperties": True}
            ):
                node.execute(state)
        events = [c[0][0] for c in mock_emit.call_args_list]
        _assert_domain_only(events)
        assert "crm_schema_validated" in events


class TestOutputFormatNodeTrace:
    def test_emits_domain_event_on_success(self):
        node = OutputFormatNode()
        state = _base_state(
            summary="Test summary",
            crm_fields="{}",
            email_draft="Test email",
            crm_validated=True,
            pii_masked=False,
            invocation_id="inv-003",
        )
        events = _emitted(node, state)
        _assert_domain_only(events)
        assert "output_formatted" in events


class TestStructuredExtractNodeTrace:
    def test_emits_domain_degraded_when_no_llm_client(self):
        node = StructuredExtractNode(llm_client=None)
        state = _base_state(annotated_transcript="テスト", invocation_id="inv-004")
        config = {"configurable": {"session_id": "corr-extract"}}
        events = _emitted(node, state)
        _assert_domain_only(events)
        assert "structured_extract_degraded" in events
