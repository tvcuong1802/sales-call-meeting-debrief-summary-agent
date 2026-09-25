"""
Tests for src/nodes/crm_schema_validate.py — CRMSchemaValidateNode.

Design doc §5.4.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.nodes.crm_schema_validate import CRMSchemaValidateNode
from src.schemas.state import initial_state


def _make_state(**overrides) -> dict:
    state = initial_state("test transcript", "sess-test")
    state.update(overrides)
    return state


def _write_tmp_schema(schema: dict) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(schema, tmp)
    tmp.close()
    return tmp.name


_SF_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "TestSchema",
    "type": "object",
    "properties": {
        "Name": {"type": ["string", "null"]},
        "Amount": {"type": ["number", "null"]},
        "StageName": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}


class TestCRMSchemaValidateNode:
    """Tests for CRMSchemaValidateNode."""

    def setup_method(self):
        self.node = CRMSchemaValidateNode()

    def _config_with_schema(self, path: str) -> dict:
        return {"crm": {"schema_path": path}}

    # ── Short-circuit ────────────────────────────────────────────────────────

    def test_short_circuit_on_error(self):
        """If state['error'] is non-empty, node returns state unchanged."""
        state = _make_state(
            crm_fields=json.dumps({"Name": "Acme"}),
            error=json.dumps({"code": "LLM_TIMEOUT", "attempt": 2}),
        )
        result = self.node.execute(state)
        # crm_validated must remain False (unchanged from initial default)
        assert result["crm_validated"] is False
        assert result["error"] != ""

    # ── Happy path ────────────────────────────────────────────────────────────

    def test_valid_crm_fields_pass(self):
        """Valid crm_fields passes validation: crm_validated=True, no errors."""
        schema_path = _write_tmp_schema(_SF_SCHEMA)
        try:
            state = _make_state(crm_fields=json.dumps({"Name": "Acme Deal", "Amount": 50000.0}))
            result = CRMSchemaValidateNode(config=self._config_with_schema(schema_path)).execute(state)
            assert result["crm_validated"] is True
            assert result["crm_validation_errors"] == ""
            assert result["error"] == ""
        finally:
            os.unlink(schema_path)

    def test_crm_fields_re_serialized(self):
        """After validation, crm_fields is re-serialized as JSON string."""
        schema_path = _write_tmp_schema(_SF_SCHEMA)
        try:
            state = _make_state(crm_fields=json.dumps({"Name": "Deal"}))
            result = CRMSchemaValidateNode(config=self._config_with_schema(schema_path)).execute(state)
            # Must be a valid JSON string
            parsed = json.loads(result["crm_fields"])
            assert isinstance(parsed, dict)
        finally:
            os.unlink(schema_path)

    # ── Validation errors ─────────────────────────────────────────────────────

    def test_invalid_field_set_to_null(self):
        """Invalid field value is set to null (graceful degradation, non-fatal)."""
        schema_path = _write_tmp_schema(_SF_SCHEMA)
        try:
            # Amount should be number, passing string
            state = _make_state(crm_fields=json.dumps({"Name": "Deal", "Amount": "not-a-number"}))
            result = CRMSchemaValidateNode(config=self._config_with_schema(schema_path)).execute(state)
            assert result["crm_validated"] is False
            assert result["error"] == ""  # field errors are NON-FATAL
            parsed = json.loads(result["crm_fields"])
            assert parsed["Amount"] is None  # set to null
        finally:
            os.unlink(schema_path)

    def test_validation_errors_recorded(self):
        """Validation errors are recorded in crm_validation_errors."""
        schema_path = _write_tmp_schema(_SF_SCHEMA)
        try:
            state = _make_state(crm_fields=json.dumps({"Amount": "bad-value"}))
            result = CRMSchemaValidateNode(config=self._config_with_schema(schema_path)).execute(state)
            assert result["crm_validation_errors"] != ""
            errors = json.loads(result["crm_validation_errors"])
            assert isinstance(errors, list)
            assert len(errors) > 0
        finally:
            os.unlink(schema_path)

    def test_additional_properties_removed(self):
        """Fields not in schema are removed when additionalProperties=false."""
        schema_path = _write_tmp_schema(_SF_SCHEMA)
        try:
            state = _make_state(crm_fields=json.dumps({"Name": "Deal", "UnknownField": "value"}))
            result = CRMSchemaValidateNode(config=self._config_with_schema(schema_path)).execute(state)
            parsed = json.loads(result["crm_fields"])
            assert "UnknownField" not in parsed
        finally:
            os.unlink(schema_path)

    # ── Schema load errors ────────────────────────────────────────────────────

    def test_missing_schema_file_sets_error(self):
        """Missing schema file sets state['error'] (structural failure)."""
        state = _make_state(crm_fields=json.dumps({"Name": "Deal"}))
        result = CRMSchemaValidateNode(config=self._config_with_schema("/nonexistent/schema.json")).execute(state)
        assert result["error"] != ""
        error = json.loads(result["error"])
        assert error["code"] == "CRM_SCHEMA_LOAD_ERROR"

    def test_empty_crm_fields_passes(self):
        """Empty crm_fields '{}' passes validation without errors."""
        schema_path = _write_tmp_schema(_SF_SCHEMA)
        try:
            state = _make_state(crm_fields="{}")
            result = CRMSchemaValidateNode(config=self._config_with_schema(schema_path)).execute(state)
            assert result["crm_validated"] is True
            assert result["error"] == ""
        finally:
            os.unlink(schema_path)

    # ── S-4 trace ─────────────────────────────────────────────────────────────

    def test_uses_bundled_salesforce_schema(self):
        """Default schema path loads salesforce_opportunity.json correctly."""
        original_dir = os.getcwd()
        try:
            os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
            state = _make_state(crm_fields=json.dumps({"Name": "Acme", "Amount": 10000.0}))
            result = self.node.execute(state)  # no config -> default path
            assert result["error"] == "" or "LOAD" not in result["error"]
        finally:
            os.chdir(original_dir)
