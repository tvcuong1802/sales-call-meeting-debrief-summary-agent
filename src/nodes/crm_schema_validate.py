"""
CRMSchemaValidateNode — CMN-C1-042 Sales Call & Meeting Debrief Summary Agent.

Post-process slot node. Validates crm_fields JSON against the injected CRM JSON Schema.
Implements S-3 custom extension for schema conformance (ADR-003).

Design doc §5.4. No LLM call. Deterministic validation transform.
"""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar

import jsonschema

from framework.nodes.function_node import FunctionNode
from framework.schemas.invocation_context import TrustLevel
from src.utils.audit import emit_trace_event
from src.services.app_config import app_config

logger = logging.getLogger(__name__)


class CRMSchemaValidateNode(FunctionNode):
    """Validate crm_fields JSON against injected CRM JSON Schema.

    Processing:
    1. Short-circuit if state["error"] non-empty
    2. Deserialize state["crm_fields"] JSON str -> dict
    3. Load CRM JSON Schema from config crm.schema_path
    4. Validate using jsonschema.validate()
    5. On field-level errors: set invalid fields to null (graceful degradation),
       record errors in crm_validation_errors, set crm_validated=False (non-fatal)
    6. Re-serialize validated dict -> state["crm_fields"]
    7. Set state["crm_validated"] = True/False
    8. S-4 trace events

    Field-level schema errors are NON-FATAL — pipeline continues.
    Missing/malformed schema file sets state["error"] (structural failure).
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # Settings reach a node through CONSTRUCTION, not a per-call argument:
        # the framework invokes execute(state) and passes none.
        super().__init__()
        self._config = dict(config) if config is not None else None

    def _cfg(self) -> dict[str, Any]:
        """The config given at construction, else config/config.yaml."""
        return self._config if self._config is not None else app_config()

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute CRM schema validation.

        Args:
            state: Current DebriefState dict.
            config: Optional framework config.

        Returns:
            Updated state dict with crm_validated, crm_validation_errors,
            and (re-serialized) crm_fields set.
        """
        try:
            # Short-circuit if upstream error
            if state.get("error"):
                return state

            # Deserialize crm_fields JSON str -> dict
            try:
                crm_dict = json.loads(state.get("crm_fields", "{}"))
            except json.JSONDecodeError as exc:
                logger.error("CRMSchemaValidateNode: crm_fields JSONDecodeError: %s", exc)
                result = dict(state)
                result["error"] = json.dumps(
                    {
                        "code": "CRM_FIELDS_DECODE_ERROR",
                        "attempt": 1,
                        "node": "CRMSchemaValidateNode",
                    }
                )
                emit_trace_event(
                    "crm_schema_validate_error",
                    {
                        "node": self.__class__.__name__,
                        "error": "JSONDecodeError",
                    },
                    state,
                )
                return result

            # Load CRM JSON Schema
            schema_path = self._get_schema_path(self._cfg())
            schema = self._load_schema(schema_path)
            if schema is None:
                # Schema load failure — structural error, set error state
                result = dict(state)
                result["error"] = json.dumps(
                    {
                        "code": "CRM_SCHEMA_LOAD_ERROR",
                        "attempt": 1,
                        "node": "CRMSchemaValidateNode",
                        "schema_path": schema_path,
                    }
                )
                emit_trace_event(
                    "crm_schema_validate_error",
                    {
                        "node": self.__class__.__name__,
                        "error": "CRM_SCHEMA_LOAD_ERROR",
                    },
                    state,
                )
                return result

            # Validate and gracefully degrade on field-level errors
            validated_dict, validation_errors = self._validate_and_degrade(crm_dict, schema)

            result = dict(state)
            result["crm_fields"] = json.dumps(validated_dict)
            result["crm_validated"] = len(validation_errors) == 0
            result["crm_validation_errors"] = json.dumps(validation_errors) if validation_errors else ""

            emit_trace_event(
                "crm_schema_validated",
                {
                    "status": "success",
                    "crm_validated": result["crm_validated"],
                    "validation_error_count": len(validation_errors),
                },
                state,
            )

            return result

        except Exception as exc:
            emit_trace_event(
                "crm_schema_validate_error",
                {
                    "node": self.__class__.__name__,
                    "error": type(exc).__name__,
                },
                state,
            )
            logger.error("CRMSchemaValidateNode error: %s: %s", type(exc).__name__, exc)
            result = dict(state)
            result["error"] = json.dumps(
                {
                    "code": "CRM_VALIDATE_ERROR",
                    "attempt": 1,
                    "node": "CRMSchemaValidateNode",
                }
            )
            return result

    def _get_schema_path(self, config: dict[str, Any] | None) -> str:
        """Extract CRM schema path from config."""
        default_path = "config/schemas/salesforce_opportunity.json"
        if config is None:
            return default_path
        configurable = config.get("configurable", {})
        crm_cfg = configurable.get("crm", config.get("crm", {}))
        schema_path: str = crm_cfg.get("schema_path", default_path)
        return schema_path

    def _load_schema(self, schema_path: str) -> dict[str, Any] | None:
        """Load JSON Schema from file path.

        Returns:
            Parsed schema dict, or None if file not found or malformed.
        """
        try:
            with open(schema_path, encoding="utf-8") as f:
                pinned: dict[str, Any] | None = json.load(f)
                return pinned
        except FileNotFoundError:
            logger.error("CRM schema file not found: %s", schema_path)
            return None
        except json.JSONDecodeError as exc:
            logger.error("CRM schema file malformed JSON: %s — %s", schema_path, exc)
            return None

    def _validate_and_degrade(
        self, crm_dict: dict[str, Any], schema: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        """Validate crm_dict against schema, nullifying invalid fields.

        Graceful degradation: invalid fields are set to null rather than
        raising an exception. Field-level errors are non-fatal.

        Pre-validation normalisation: for string enum fields, attempts
        case-insensitive matching before calling jsonschema. This prevents
        silent null for LLM outputs like "closed won" when the schema
        requires "Closed Won" (issue #30).

        Args:
            crm_dict: Deserialized CRM fields dict.
            schema: JSON Schema dict.

        Returns:
            Tuple of (validated_dict, list_of_error_messages).
        """
        validation_errors: list[str] = []
        validated = dict(crm_dict)

        # Check additionalProperties — remove fields not in schema
        schema_props = schema.get("properties", {})
        additional_props_allowed = schema.get("additionalProperties", True)

        if not additional_props_allowed and schema_props:
            extra_keys = [k for k in list(validated.keys()) if k not in schema_props]
            for key in extra_keys:
                validation_errors.append(f"Additional property not allowed: {key!r}")
                validated.pop(key, None)

        # Validate each property against its type constraint
        for key, prop_schema in schema_props.items():
            if key not in validated:
                continue
            value = validated[key]
            if value is None:
                continue  # null is always acceptable for nullable fields

            # Pre-normalise string enum values (case-insensitive match)
            # Prevents silent null when LLM returns "closed won" for "Closed Won"
            value = self._normalise_enum_value(value, prop_schema)
            validated[key] = value

            try:
                jsonschema.validate(instance=value, schema=prop_schema)
            except jsonschema.ValidationError as exc:
                validation_errors.append(f"Field {key!r}: {exc.message}")
                validated[key] = None  # graceful degradation — set to null

        return validated, validation_errors

    def _normalise_enum_value(self, value: Any, prop_schema: dict[str, Any]) -> Any:
        """Case-insensitive normalisation for string enum fields.

        If the schema defines an enum, and the value is a string that does not
        match any enum member exactly but does match case-insensitively, the
        canonical enum member is returned. Otherwise the original value is returned
        unchanged.

        Args:
            value: Field value from LLM output.
            prop_schema: JSON Schema property definition dict.

        Returns:
            Normalised value (canonical enum string if matched, original otherwise).
        """
        enum_values = prop_schema.get("enum")
        if not enum_values or not isinstance(value, str):
            return value

        # Exact match — no normalisation needed
        if value in enum_values:
            return value

        # Case-insensitive match against string enum members
        value_lower = value.lower()
        for candidate in enum_values:
            if isinstance(candidate, str) and candidate.lower() == value_lower:
                logger.info("Enum case normalised: %r → %r", value, candidate)
                return candidate

        # No match found — return original (will fail validation → null)
        return value
