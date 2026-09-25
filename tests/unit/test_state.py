"""
Tests for src/state.py — DebriefState TypedDict and initial_state() factory.
"""

from src.schemas.state import initial_state


class TestInitialState:
    """Tests for initial_state() factory."""

    def test_returns_debrief_state(self):
        """initial_state() returns a dict that satisfies DebriefState contract."""
        state = initial_state("some transcript", "session-001")
        # TypedDict is a dict at runtime
        assert isinstance(state, dict)

    def test_input_fields_set(self):
        """raw_transcript and session_id are set from arguments."""
        state = initial_state("hello world", "sess-123")
        assert state["raw_transcript"] == "hello world"
        assert state["session_id"] == "sess-123"

    def test_all_fields_present(self):
        """All required state fields are present."""
        state = initial_state("transcript", "session")
        expected_keys = {
            "raw_transcript",
            "session_id",
            "cleaned_transcript",
            "annotated_transcript",
            "pii_masked",
            "pii_mask_count",
            "summary",
            "cannot_summarize",
            "crm_fields",
            "email_draft",
            "crm_validated",
            "crm_validation_errors",
            "output_json",
            "error",
            "retry_count",
            "invocation_id",
            "audit_input_hash",
            "audit_output_hash",
        }
        assert expected_keys == set(state.keys())

    def test_string_defaults_are_empty(self):
        """String fields default to empty string (not None)."""
        state = initial_state("t", "s")
        for key in (
            "cleaned_transcript",
            "annotated_transcript",
            "summary",
            "email_draft",
            "crm_validation_errors",
            "output_json",
            "error",
            "invocation_id",
            "audit_input_hash",
            "audit_output_hash",
        ):
            assert state[key] == "", f"Expected '' for {key}, got {state[key]!r}"

    def test_crm_fields_default_is_empty_json_object(self):
        """crm_fields defaults to '{}' (JSON-safe empty object, not empty string)."""
        state = initial_state("t", "s")
        assert state["crm_fields"] == "{}"

    def test_bool_defaults(self):
        """Boolean fields default to False."""
        state = initial_state("t", "s")
        assert state["pii_masked"] is False
        assert state["crm_validated"] is False

    def test_int_defaults(self):
        """Integer fields default to 0."""
        state = initial_state("t", "s")
        assert state["pii_mask_count"] == 0
        assert state["retry_count"] == 0

    def test_all_fields_are_primitive_types(self):
        """All fields are primitives (str, bool, int) — no dicts, Pydantic, etc."""
        state = initial_state("transcript", "session")
        for key, value in state.items():
            assert isinstance(value, (str, bool, int)), f"Field {key!r} has non-primitive type {type(value).__name__!r}"

    def test_crm_fields_is_json_deserializable(self):
        """crm_fields default value is valid JSON."""
        import json

        state = initial_state("t", "s")
        parsed = json.loads(state["crm_fields"])
        assert parsed == {}

    def test_factory_called_twice_returns_independent_dicts(self):
        """Two calls to initial_state() return independent dicts (no shared state)."""
        s1 = initial_state("t1", "s1")
        s2 = initial_state("t2", "s2")
        s1["summary"] = "modified"
        assert s2["summary"] == ""  # unchanged

    def test_empty_transcript_allowed(self):
        """initial_state() accepts empty string for raw_transcript (validation is node-level)."""
        state = initial_state("", "session-empty")
        assert state["raw_transcript"] == ""
