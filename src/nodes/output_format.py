"""
OutputFormatNode — CMN-C1-042 Sales Call & Meeting Debrief Summary Agent.

Post-process slot node. Formats final output JSON and implements S-3 content
safety and internal data redaction gate.

Design doc §5.5. No LLM call. Deterministic formatting + security gate.

IMPORTANT: This node runs its S-3 gate EVEN IF state["error"] is non-empty.
No unsafe partial output may escape the pipeline (design doc §6.3, §8.3).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, ClassVar

from framework.errors import SecurityViolationError
from framework.nodes.function_node import FunctionNode
from framework.schemas.invocation_context import TrustLevel
from src.utils.audit import emit_trace_event
from src.services.app_config import app_config

logger = logging.getLogger(__name__)

# S-3 credential patterns (design doc §8.3)
_CREDENTIAL_PATTERNS = [
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{20,}"),  # Bearer tokens
    re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{19,}"),  # OpenAI-style API keys
    re.compile(r"AKIA[A-Z0-9]{16}"),  # AWS access keys
    re.compile(r"['\"]?[A-Za-z0-9\-_]{32,}['\"]?\s*:\s*['\"][A-Za-z0-9\-_]{20,}['\"]"),  # connection string
    re.compile(r"(?i)password\s*[=:]\s*\S{8,}"),  # password= patterns
    re.compile(r"(?i)api[_-]?key\s*[=:]\s*\S{8,}"),  # api_key= patterns
]

# S-3 prompt injection markers (design doc §8.3)
_INJECTION_PATTERNS = [
    re.compile(r"\[INST\]"),
    re.compile(r"<\|system\|>"),
    re.compile(r"<\|user\|>"),
    re.compile(r"<\|assistant\|>"),
    re.compile(r"###\s*System"),
    re.compile(r"###\s*Instruction"),
]


# Section labels. Everything the reader sees that is not their own content comes from
# here, so a Japanese debrief is a Japanese document rather than English headings wrapped
# around Japanese text. Closed set: four headings and two empty-state lines.
_CHROME: dict[str, dict[str, str]] = {
    "en": {
        "degraded": (
            "## No debrief was produced\n\n"
            "The summary could not be generated for this request, so nothing below was "
            "read from your transcript. Please try again shortly."
        ),
        "summary": "## Debrief",
        "crm": "## CRM fields",
        "email": "## Follow-up email draft",
        "issues": "## Needs attention",
        "no_crm": "_No CRM fields could be filled from this transcript._",
        "no_email": "_No follow-up email was drafted._",
    },
    "ja": {
        "degraded": (
            "## 要約を作成できませんでした\n\n"
            "この依頼では要約を生成できなかったため、以下の内容は議事録から読み取ったものではありません。"
            "しばらくしてから再度お試しください。"
        ),
        "summary": "## 商談サマリ",
        "crm": "## CRM 項目",
        "email": "## フォローアップメール案",
        "issues": "## 要確認",
        "no_crm": "_この議事録から CRM 項目を特定できませんでした。_",
        "no_email": "_フォローアップメールは作成されませんでした。_",
    },
}


def _as_markdown(output: dict[str, Any], language: str) -> str:
    """Render the debrief for a person. The JSON stays untouched for callers that want it."""
    chrome = _CHROME["ja"] if str(language).lower() == "ja" else _CHROME["en"]

    # Say when nothing was read. Without this the reply was four empty sections --
    # "No CRM fields could be filled from this transcript" reads as a finding ABOUT the
    # transcript, not as "the model never ran", and a reader has no way to tell the two
    # apart. That is the expensive shape: not a crash, a confident wrong answer.
    if not str(output.get("summary", "")).strip() and not (output.get("crm_fields") or {}):
        # Bilingual when nothing was read: this is the branch where the language could
        # not be determined either, and a notice the reader cannot read is the same as
        # no notice.
        if not str(language).strip():
            return _CHROME["en"]["degraded"] + "\n\n" + _CHROME["ja"]["degraded"]
        return chrome["degraded"]

    parts: list[str] = []

    summary = str(output.get("summary", "")).strip()
    if summary:
        parts.append(f"{chrome['summary']}\n\n{summary}")

    crm = output.get("crm_fields") or {}
    if isinstance(crm, dict) and crm:
        rows = "\n".join(f"- **{k}**: {v}" for k, v in crm.items())
        parts.append(f"{chrome['crm']}\n\n{rows}")
    else:
        parts.append(f"{chrome['crm']}\n\n{chrome['no_crm']}")

    email = str(output.get("email_draft", "")).strip()
    parts.append(f"{chrome['email']}\n\n{email}" if email else f"{chrome['email']}\n\n{chrome['no_email']}")

    # Validation errors were recorded in meta and never shown. They are the reader's
    # signal that a CRM value should not be pasted anywhere yet.
    meta = output.get("meta") or {}
    issues = [str(e) for e in (meta.get("crm_validation_errors") or []) if str(e).strip()]
    if issues:
        listed = "\n".join(f"- {e}" for e in issues)
        parts.append(f"{chrome['issues']}\n\n{listed}")

    return "\n\n".join(parts)


class OutputFormatNode(FunctionNode):
    """Format final output JSON with S-3 content safety gate.

    Processing:
    1. S-3 output gate (ALWAYS runs, even if state["error"] non-empty)
    2. Assemble final output dict from state fields
    3. Serialize to output_json
    4. Compute audit_output_hash (SHA-256 of output_json)
    5. S-4 trace events

    The S-3 gate scans summary and email_draft for credential patterns and
    prompt injection markers. SecurityViolationError is raised (hard stop)
    on detection — never suppressed.
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
        """Execute output formatting with S-3 gate.

        NOTE: S-3 gate runs EVEN IF state["error"] is non-empty.

        Args:
            state: Current DebriefState dict.

        Returns:
            Updated state dict with output_json and audit_output_hash set.

        Raises:
            SecurityViolationError: If S-3 gate detects credential or injection pattern.
        """
        try:
            # Work on a copy — _security_gate_output may mutate summary/email_draft
            # for redaction. Copy first so the caller's state dict is never mutated.
            result = dict(state)

            # S-3 output gate — always runs (design doc §6.3, §8.3).
            # Private node-level helper (NOT an override of FunctionNode's @final
            # _security_gate_output, whose signature is (state)); the framework
            # @final S-3 gate still fires per-node as defence in depth (CR-0930).
            self._scan_output(result)

            # Assemble final output dict
            error_in_state = result.get("error", "")
            error_dict = json.loads(error_in_state) if error_in_state else None

            # Deserialize crm_fields for output (gracefully handle decode errors)
            crm_fields_str = result.get("crm_fields", "{}")
            try:
                crm_fields_dict = json.loads(crm_fields_str)
            except json.JSONDecodeError:
                crm_fields_dict = {}

            # Parse crm_validation_errors for meta (empty list if none)
            crm_val_errors_str = result.get("crm_validation_errors", "")
            # list[str], read from the producer: CRMSchemaValidateNode builds
            # `validation_errors: list[str]` and only ever appends f-strings to it
            # (src/nodes/crm_schema_validate.py:191-220), then json.dumps() it into this
            # state field.
            crm_val_errors: list[str] = json.loads(crm_val_errors_str) if crm_val_errors_str else []

            output_dict: dict[str, Any] = {
                "summary": result.get("summary", ""),
                "crm_fields": crm_fields_dict,
                "email_draft": result.get("email_draft", ""),
                "meta": {
                    "crm_validated": result.get("crm_validated", False),
                    "pii_masked": result.get("pii_masked", False),
                    "invocation_id": result.get("invocation_id", ""),
                    "crm_validation_errors": crm_val_errors,
                },
            }

            if error_dict is not None:
                output_dict["error"] = error_dict

            # Serialize and hash
            # The reader gets Markdown; the machine keeps the JSON. Marketplace shows
            # `output` verbatim in a chat surface, so a JSON blob there is a reply the
            # customer has to parse by eye -- three different things (a debrief, CRM
            # values, an email draft) run together inside one string.
            result["output_markdown"] = _as_markdown(output_dict, result.get("output_language", "en"))
            output_json = json.dumps(output_dict, ensure_ascii=False)
            audit_output_hash = hashlib.sha256(output_json.encode("utf-8")).hexdigest()

            result["output_json"] = output_json
            result["audit_output_hash"] = audit_output_hash

            emit_trace_event(
                "output_formatted",
                {
                    "status": "success",
                    "audit_output_hash": audit_output_hash,
                    "has_error": error_dict is not None,
                },
                state,
            )

            return result

        except SecurityViolationError:
            # S-3 hard stop — re-raise, never suppress
            emit_trace_event(
                "output_format_error",
                {
                    "node": self.__class__.__name__,
                    "error": "SecurityViolationError",
                },
                state,
            )
            raise

        except Exception as exc:
            emit_trace_event(
                "output_format_error",
                {
                    "node": self.__class__.__name__,
                    "error": type(exc).__name__,
                },
                state,
            )
            logger.error("OutputFormatNode error: %s: %s", type(exc).__name__, exc)
            result = dict(state)
            result["error"] = json.dumps(
                {
                    "code": "OUTPUT_FORMAT_ERROR",
                    "attempt": 1,
                    "node": "OutputFormatNode",
                }
            )
            return result

    def _scan_output(self, state: dict[str, Any]) -> None:
        """S-3 content safety scan (node-level private helper).

        Named ``_scan_output`` (not ``_security_gate_output``) to avoid colliding
        with FunctionNode's @final S-3 gate; called inline in execute() so it
        scans the assembled output dict at the right ordering point (CR-0930).

        Scans summary and email_draft for:
        1. Credential patterns (API keys, Bearer tokens, connection strings) — hard stop
        2. Prompt injection markers ([INST], <|system|>, etc.) — hard stop
        3. Internal data redaction via config security.redact_patterns

        Args:
            state: Current DebriefState dict.

        Raises:
            SecurityViolationError: If credential or injection pattern detected.
        """
        summary = state.get("summary", "")
        email_draft = state.get("email_draft", "")
        # Also scan crm_fields — LLM could inject credential patterns into field values
        crm_fields_raw = state.get("crm_fields", "{}")
        combined = f"{summary}\n{email_draft}\n{crm_fields_raw}"

        # Check credential patterns
        for pattern in _CREDENTIAL_PATTERNS:
            if pattern.search(combined):
                raise SecurityViolationError(
                    f"S-3 gate: credential pattern detected in output (pattern: {pattern.pattern!r})"
                )

        # Check prompt injection markers
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(combined):
                raise SecurityViolationError(
                    f"S-3 gate: prompt injection marker detected in output (pattern: {pattern.pattern!r})"
                )

        # Apply redact_patterns from config
        redact_patterns = self._get_redact_patterns(self._cfg())
        if redact_patterns:
            # Redaction mutates state in-place (summary and email_draft)
            redacted_summary = summary
            redacted_email = email_draft
            redacted_any = False

            for raw_pattern in redact_patterns:
                try:
                    compiled = re.compile(raw_pattern)
                    new_summary = compiled.sub("[REDACTED]", redacted_summary)
                    new_email = compiled.sub("[REDACTED]", redacted_email)
                    if new_summary != redacted_summary or new_email != redacted_email:
                        redacted_any = True
                    redacted_summary = new_summary
                    redacted_email = new_email
                except re.error as exc:
                    logger.warning("Invalid redact pattern %r: %s — skipping", raw_pattern, exc)

            if redacted_any:
                state["summary"] = redacted_summary
                state["email_draft"] = redacted_email
                emit_trace_event(
                    "s3_redaction_applied",
                    {
                        "node": self.__class__.__name__,
                        "patterns_applied": len(redact_patterns),
                    },
                    state,
                )

    def _get_redact_patterns(self, config: dict[str, Any] | None) -> list[str]:
        """Extract redact_patterns list from config."""
        if config is None:
            return []
        configurable = config.get("configurable", {})
        security_cfg = configurable.get("security", config.get("security", {}))
        patterns: list[str] = security_cfg.get("redact_patterns", [])
        return patterns
