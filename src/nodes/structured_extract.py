"""
StructuredExtractNode — CMN-C1-042 Sales Call & Meeting Debrief Summary Agent.

Main slot node. Single-pass LLM extraction producing all three output keys.

Design doc §5.3. ADR-001 (single-pass constraint — DO NOT add LLM calls).
ADR-002 (L1 direct inheritance).

Part 1: Core extraction + hardcoded BANT baseline prompt.
Part 2: YAML dynamic injection — methodology (BANT/MEDDIC), Keigo branches,
        CRM schema field injection.
Part 3: Retry wrapper wiring + S-4 trace events on retry.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel, ValidationError

from framework.nodes.function_node import FunctionNode
from framework.schemas.invocation_context import TrustLevel
from src.services.agent_scope import INTAKE_POLICY
from src.services.input_intake import understand_input
from src.services.llm_provider import build_llm_client
from src.utils.audit import emit_trace_event
from src.retry import (
    ValidationRetryError,
    RateLimitRetryError,
    TimeoutRetryError,
    call_with_retry_typed,
)

logger = logging.getLogger(__name__)

# ── Pydantic output schema (design doc §1.3, §5.3) ───────────────────────────


class DebriefOutput(BaseModel):
    """Structured output schema for single-pass LLM extraction.

    Used ONLY inside StructuredExtractNode for LLM structured-output parsing.
    Instantiated, unpacked to primitives, then discarded — never stored in state.
    """

    summary: str
    crm_fields: dict[str, Any]
    email_draft: str


# ── Static prompt blocks (language-agnostic, not in YAML) ────────────────────

_ROLE_BLOCK = (
    "You are a structured extraction engine for B2B sales meeting transcripts. "
    "Respond ONLY with a valid JSON object that exactly matches the required schema. "
    "Do not include any text, explanation, or markdown outside the JSON object."
)

_ANTI_HALLUCINATION_BLOCK = (
    "CRITICAL RULE: You MUST output null for any CRM field not explicitly stated "
    "in the transcript. Do NOT infer, estimate, or fabricate values. "
    "If a field was not discussed, its value must be null."
)

# Default prompt file paths (relative to working directory)
_DEFAULT_PROMPTS_EN = "config/prompts.yaml"
_DEFAULT_PROMPTS_JA = "config/prompts_ja.yaml"
_DEFAULT_CONFIG = "config/config.yaml"


# ── Prompt loading helpers ────────────────────────────────────────────────────


def _load_yaml(path: str) -> dict[str, Any]:
    """Load and parse a YAML file. Raises FileNotFoundError or yaml.YAMLError."""
    with open(path, encoding="utf-8") as f:
        pinned: dict[str, Any] = yaml.safe_load(f)
        return pinned


def _build_crm_fields_block(schema: dict[str, Any]) -> str:
    """Build the CRM fields section for the prompt from a loaded JSON Schema.

    Extracts property names and descriptions from the schema's 'properties' key
    and formats them as a numbered field list for the LLM.

    Args:
        schema: Parsed CRM JSON Schema dict.

    Returns:
        Formatted string listing each CRM field with its description.
    """
    properties = schema.get("properties", {})
    if not properties:
        return "Extract all relevant CRM fields from the transcript."

    lines = ["CRM fields to extract (output null for any field not discussed):"]
    for field_name, field_def in properties.items():
        description = field_def.get("description", "")
        field_type = field_def.get("type", "")
        # Normalize type list to readable string
        if isinstance(field_type, list):
            field_type = " | ".join(t for t in field_type if t != "null")
        desc_str = f" — {description}" if description else ""
        lines.append(f"- {field_name} ({field_type}){desc_str}")
    return "\n".join(lines)


def _build_output_schema_block(schema: dict[str, Any]) -> str:
    """Build the output JSON schema example block from the CRM schema.

    Args:
        schema: Parsed CRM JSON Schema dict.

    Returns:
        Formatted string showing the exact JSON structure the LLM must return.
    """
    properties = schema.get("properties", {})
    crm_fields_example = {name: None for name in properties} if properties else {}
    output_example = {
        "summary": "<structured meeting summary: discussion points, objections, next actions, deal status>",
        "crm_fields": crm_fields_example,
        "email_draft": "<Subject: ...\\n\\n<email body>",
    }
    return "You must respond with ONLY this JSON structure (no other text):\n" + json.dumps(
        output_example, ensure_ascii=False, indent=2
    )


def _build_prompt(
    annotated_transcript: str,
    methodology_block: str,
    email_block: str,
    crm_fields_block: str,
    output_schema_block: str,
) -> str:
    """Build the instruction-first prompt for single-pass extraction.

    Structure (instruction-first per PM alignment):
    1. Role block
    2. Anti-hallucination constraint
    3. Methodology block (BANT or MEDDIC — from prompts.yaml)
    4. CRM fields block (derived from crm.schema_path JSON Schema)
    5. Email block (formality branch — from prompts_ja.yaml)
    6. Output schema block
    7. Transcript

    Args:
        annotated_transcript: Cleaned + nuance-annotated transcript.
        methodology_block: Selected methodology instructions from prompts.yaml.
        email_block: Selected email formality instructions from prompts_ja.yaml.
        crm_fields_block: CRM field list derived from the loaded JSON Schema.
        output_schema_block: Output JSON structure example.

    Returns:
        Full prompt string for LLM call.
    """
    parts = [
        _ROLE_BLOCK,
        "",
        _ANTI_HALLUCINATION_BLOCK,
        "",
        methodology_block,
        "",
        crm_fields_block,
        "",
        email_block,
        "",
        output_schema_block,
        "",
        "--- TRANSCRIPT BEGIN ---",
        annotated_transcript,
        "--- TRANSCRIPT END ---",
    ]
    return "\n".join(parts)


# ── Node implementation ───────────────────────────────────────────────────────


def _script_language(text: str) -> str | None:
    """Fallback only: what the characters are, when the model returns no usable code."""
    if any("\u3040" <= c <= "\u30ff" or "\u4e00" <= c <= "\u9fff" for c in text or ""):
        return "ja"
    return "en" if text and text.strip() else None


class StructuredExtractNode(FunctionNode):
    """Single-pass LLM extraction node (main slot).

    Produces summary, crm_fields (JSON str), email_draft from one LLM call.

    Part 2 additions:
    - Loads config/config.yaml at __init__ time (overridable via config_override)
    - Loads prompts.yaml / prompts_ja.yaml based on agent.output_language
    - Selects methodology block via config.methodology (bant | meddic)
    - Selects email block via config.email.formality_level (formal | semi_formal | casual)
    - Parses CRM JSON Schema from crm.schema_path → injects field names into prompt

    Part 3 additions:
    - Wraps LLM call with call_with_retry_typed() from src/retry.py
    - Uses config.llm.retry.max_attempts and config.llm.retry.backoff_factor
    - Emits node_retry trace event on each retry (attempt, error_type, node_name)
    - ADR-001 still holds: exactly 1 logical extraction attempt per user call;
      retry is internal to the node and transparent to the graph

    LLM client contract:
        llm_client.invoke(prompt: str) -> str
        Returns raw LLM response string (may be JSON or freeform).
        The node handles parsing; the client does NOT parse.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def __init__(
        self,
        llm_client: Any = None,
        config_path: str = _DEFAULT_CONFIG,
        config_override: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize node, load config and prompt files.

        Args:
            llm_client: LLM client with .invoke(prompt: str) -> str interface.
            config_path: Path to config/config.yaml. Default: "config/config.yaml".
            config_override: Optional dict to override config values (for tests).
        """
        try:
            super().__init__(**kwargs)
        except TypeError:
            pass
        self._llm_client = llm_client
        self._agent_config = self._load_agent_config(config_path, config_override)
        self._methodology_block, self._email_block, self._crm_fields_block, self._output_schema_block = (
            self._load_prompt_components(self._agent_config)
        )

    def _load_agent_config(self, config_path: str, config_override: dict[str, Any] | None) -> dict[str, Any]:
        """Load agent config from YAML file with optional override."""
        try:
            cfg = _load_yaml(config_path)
        except FileNotFoundError:
            logger.warning("Config file not found: %s — using empty config", config_path)
            cfg = {}
        if config_override:
            cfg.update(config_override)
        return cfg

    def _load_prompt_components(self, cfg: dict[str, Any]) -> tuple[str, str, str, str]:
        """Load and select methodology block, email block, CRM fields block.

        Args:
            cfg: Loaded agent config dict.

        Returns:
            Tuple of (methodology_block, email_block, crm_fields_block, output_schema_block).
        """
        output_language = cfg.get("agent", {}).get("output_language") or "en"
        methodology = cfg.get("methodology", "bant").lower()
        formality = cfg.get("email", {}).get("formality_level", "formal").lower()
        self._formality = formality
        # What the operator DECLARED, if anything. "ja" used to be the hardcoded default
        # and it was resolved once at construction, so an English transcript came back as
        # a Japanese debrief on every request -- measured against real Azure 2026-08-28.
        self._declared_language = cfg.get("agent", {}).get("output_language")
        schema_path = cfg.get("crm", {}).get("schema_path", "config/schemas/salesforce_opportunity.json")

        # Load methodology block from prompts.yaml (EN) — language-agnostic methodology
        methodology_block = self._load_methodology_block(methodology)

        # Load email block from prompts_ja.yaml (JP) or prompts.yaml (EN)
        email_block = self._load_email_block(formality, output_language)

        # Load CRM schema and build field list
        crm_schema = self._load_crm_schema(schema_path)
        crm_fields_block = _build_crm_fields_block(crm_schema)
        output_schema_block = _build_output_schema_block(crm_schema)

        return methodology_block, email_block, crm_fields_block, output_schema_block

    def _load_methodology_block(self, methodology: str) -> str:
        """Load methodology instructions from prompts.yaml.

        Args:
            methodology: "bant" or "meddic"

        Returns:
            Methodology instructions string. Falls back to hardcoded BANT if file missing.
        """
        try:
            prompts = _load_yaml(_DEFAULT_PROMPTS_EN)
            methods = prompts.get("methodologies", {})
            if methodology in methods:
                instructions: str = methods[methodology].get("instructions", "").strip()
                return instructions
            logger.warning("Methodology %r not found in prompts.yaml — falling back to bant", methodology)
            bant_instructions: str = methods.get("bant", {}).get("instructions", "").strip()
            return bant_instructions
        except FileNotFoundError:
            logger.warning("prompts.yaml not found — using hardcoded BANT baseline")
            return (
                "Extract BANT sales qualification fields from the transcript into crm_fields. "
                "Output null for any field not explicitly discussed."
            )

    def _load_email_block(self, formality: str, output_language: str) -> str:
        """Load email formality instructions from prompts_ja.yaml (JP) or prompts.yaml (EN).

        Args:
            formality: "formal", "semi_formal", or "casual"
            output_language: "ja" or "en"

        Returns:
            Email instructions string. Falls back gracefully if file or key missing.
        """
        prompts_file = _DEFAULT_PROMPTS_JA if output_language == "ja" else _DEFAULT_PROMPTS_EN
        try:
            prompts = _load_yaml(prompts_file)
            formality_branches = prompts.get("email_formality", {})
            if formality in formality_branches:
                branch: str = formality_branches[formality].get("instructions", "").strip()
                return branch
            logger.warning("Formality %r not in %s — falling back to formal", formality, prompts_file)
            formal_branch: str = formality_branches.get("formal", {}).get("instructions", "").strip()
            return formal_branch
        except FileNotFoundError:
            logger.warning("%s not found — using fallback email instruction", prompts_file)
            return (
                "Write a professional follow-up email based on the meeting summary. "
                "Include subject and body in the email_draft field."
            )

    def _load_crm_schema(self, schema_path: str) -> dict[str, Any]:
        """Load CRM JSON Schema from file.

        Args:
            schema_path: Path to the CRM JSON Schema file.

        Returns:
            Parsed schema dict. Returns empty dict on file-not-found or JSON error.
        """
        try:
            with open(schema_path, encoding="utf-8") as f:
                pinned: dict[str, Any] = json.load(f)
                return pinned
        except FileNotFoundError:
            logger.warning("CRM schema not found: %s — prompt will have generic field instruction", schema_path)
            return {}
        except json.JSONDecodeError as exc:
            logger.error("CRM schema malformed JSON: %s — %s", schema_path, exc)
            return {}

    def _on_retry(self, attempt: int, error_type: str, node_name: str, state: dict[str, Any] | None = None) -> None:
        """Callback invoked by call_with_retry_typed on each retry attempt.

        Emits node_retry S-4 trace event per design doc §9 / Part 3 spec.

        `state` is threaded so the audit record carries trace_id / correlation_id /
        session_id. It is keyword-defaulted because `call_with_retry_typed` owns the
        3-argument callback contract (src/retry.py); execute() binds the live state via
        a closure rather than widening that shared signature.

        Args:
            attempt: Retry attempt number (1-based).
            error_type: Error type string (e.g. "LLM_SCHEMA_FAILURE").
            node_name: Node name string.
            state: Graph state, for audit correlation fields.
        """
        emit_trace_event(
            "node_retry",
            {
                "node": node_name,
                "attempt": attempt,
                "error_type": error_type,
            },
            state,
        )

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute single-pass LLM extraction with retry wrapper.

        ADR-001: Exactly 1 logical extraction attempt per user call.
        Retry is internal — the graph sees one execute() call regardless of retries.

        Args:
            state: Current DebriefState dict. Reads annotated_transcript.
            config: Optional framework config.

        Returns:
            Updated state with summary, crm_fields (JSON str), email_draft set.
            On failure: state["error"] set with structured error code.
        """
        try:
            # Short-circuit if upstream error
            if state.get("error"):
                return state

            # Guard: no LLM client — FAIL-CLOSED to a safe, empty advisory result
            # (never status:error). The debrief summary is an LLM-derived artifact;
            # with no LLM available the node returns an empty summary and a
            # cannot_summarize marker instead of raising, so the pipeline still
            # completes successfully (e.g. the STG smoke path, which runs with no
            # LLM secret). Callers see an explicit "no summary" rather than a crash.
            # The Marketplace runner constructs the graph as `agent_cls()` with no
            # arguments, so a client injected at construction is never supplied there and
            # this node returned an empty summary on every real request -- fail-closed,
            # reported as success. Build one from the invocation's own secrets instead,
            # here inside execute() where a context exists. Never assigned to self: node
            # instances are shared across concurrent invocations.
            llm_client = self._llm_client or build_llm_client(dict(state))
            if llm_client is None:
                logger.warning("StructuredExtractNode: llm_client is None — returning empty advisory (fail-closed)")
                emit_trace_event(
                    "structured_extract_degraded",
                    {
                        "node": self.__class__.__name__,
                        "reason": "llm_client_none",
                    },
                    state,
                )
                return {
                    "summary": "",
                    "crm_fields": "{}",
                    "cannot_summarize": True,
                    # Empty, not "en": nothing was read, so no language was determined.
                    # The renderer prints both notices on that branch.
                    "output_language": "",
                }

            # The debrief is for whoever wrote the notes, so it follows their language.
            # An explicit output_language in config still wins -- a deployment pinned to
            # one language keeps it. Otherwise the intake call reads the transcript, which
            # answers "what language does this reader want" rather than "what characters
            # are these": a transcript typed in romanised Japanese is entirely Latin.
            annotated_transcript = state.get("annotated_transcript", "")
            # A message with no letters and no kana -- punctuation, digits -- carries no
            # transcript and no language. Calling the model anyway spends a request and
            # gets back filler wrapped in whichever language it defaults to, which then
            # prints a single-language notice on the one path where the language is least
            # knowable. Skipping is both cheaper and more honest.
            # Judge the ORIGINAL message, not the annotated transcript: the nuance node
            # stamps markers onto it, so an input of pure punctuation arrives here
            # carrying Latin letters that were never the user's.
            declared = str(getattr(self, "_declared_language", "") or "").strip().lower()[:2]
            if declared in ("en", "ja"):
                language, language_source = declared, "declared"
            else:
                intake = understand_input(
                    annotated_transcript or state.get("user_input", ""),
                    llm_client,
                    policy=INTAKE_POLICY,
                    script_language=_script_language,
                )
                language, language_source = intake["answer_language"], intake["source"]
            emit_trace_event(
                "debrief_language_resolved",
                {"language": language, "source": language_source},
                state,
            )
            email_block = self._load_email_block(getattr(self, "_formality", "formal"), language)
            # Name EVERY reader-visible field, not just the summary. Saying "write the
            # summary and the email draft in Japanese" left the CRM values to the model's
            # own default, and it wrote English descriptions into them -- so a Japanese
            # debrief came back mixed about one run in three, and the liability notice
            # flapped between Japanese and bilingual on identical input.
            #
            # And say what carries over untranslated. Field NAMES are the customer's
            # schema keys (StageName, NextStep, CloseDate); translating those would break
            # the record they are written into, which is a worse failure than the one
            # being fixed.
            written_in = "Japanese" if language == "ja" else "English"
            language_instruction = (
                f"Write every part the reader sees in {written_in}: the summary, the CRM "
                f"field VALUES, and the email draft. The CRM field NAMES are schema keys "
                f"and stay exactly as given, untranslated. Proper nouns -- company and "
                f"product names -- also stay as they appear in the transcript. Nothing "
                f"else is written in another language."
            )
            prompt = _build_prompt(
                annotated_transcript,
                methodology_block=language_instruction + "\n" + self._methodology_block,
                email_block=email_block,
                crm_fields_block=self._crm_fields_block,
                output_schema_block=self._output_schema_block,
            )

            # Build retry config from agent config (config.yaml llm.retry structure)
            retry_config = {
                "llm": self._agent_config.get("llm", {}),
            }

            # Wrap LLM call in retry wrapper (Part 3)
            # _llm_fn raises ValidationRetryError on parse failure so retry fires correctly
            result_state = call_with_retry_typed(
                self._make_llm_fn(prompt, state, llm_client),
                state,
                config=retry_config,
                on_retry=lambda a, e, n: self._on_retry(a, e, n, state),
            )

            result_state = dict(result_state)
            result_state["output_language"] = language
            if result_state.get("error"):
                emit_trace_event(
                    "structured_extract_error",
                    {
                        "node": self.__class__.__name__,
                        "error": "retry_exhausted",
                        "error_code": json.loads(result_state["error"]).get("code", ""),
                    },
                    state,
                )
                return result_state

            emit_trace_event(
                "structured_extracted",
                {
                    "status": "success",
                    "invocation_id": state.get("invocation_id", ""),
                    "crm_fields_count": len(json.loads(result_state.get("crm_fields", "{}"))),
                },
                state,
            )

            return result_state

        except Exception as exc:
            logger.error("StructuredExtractNode unexpected error: %s: %s", type(exc).__name__, exc)
            emit_trace_event(
                "structured_extract_error",
                {
                    "node": self.__class__.__name__,
                    "error": type(exc).__name__,
                },
                state,
            )
            result = dict(state)
            result["error"] = json.dumps(
                {
                    "code": "LLM_SCHEMA_FAILURE",
                    "attempt": 1,
                    "node": "StructuredExtractNode",
                }
            )
            return result

    def _make_llm_fn(
        self,
        prompt: str,
        original_state: dict[str, Any],
        llm_client: Any = None,
    ) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """Build the LLM callable for call_with_retry_typed.

        Returns a function that:
        - Invokes the LLM with the built prompt (ADR-001: exactly one invoke call)
        - Parses and validates the response via Pydantic
        - Raises ValidationRetryError on parse failure (triggers retry)
        - Returns updated state dict on success

        The prompt is rebuilt using the (possibly modified) state's
        annotated_transcript on each retry attempt — enabling the retry wrapper
        to inject STRICT_PROMPT_SUFFIX via state["annotated_transcript"].
        """
        node = self  # capture for closure

        # Resolved once, outside the closure: the retry helper may call this several
        # times and rebuilding a client per attempt would hide a credential problem
        # behind a retry loop.
        client = llm_client or self._llm_client

        def llm_fn(state: dict[str, Any]) -> dict[str, Any]:
            # Rebuild prompt if transcript was modified by retry wrapper
            current_transcript = state.get("annotated_transcript", "")
            if current_transcript != original_state.get("annotated_transcript", ""):
                # Retry modified the transcript (strict suffix or truncation)
                current_prompt = _build_prompt(
                    current_transcript,
                    methodology_block=node._methodology_block,
                    email_block=node._email_block,
                    crm_fields_block=node._crm_fields_block,
                    output_schema_block=node._output_schema_block,
                )
            else:
                current_prompt = prompt

            # Single LLM call (ADR-001 — exactly one invoke per attempt)
            try:
                # The client resolved in execute() for THIS invocation, not
                # node._llm_client: the Marketplace runner constructs the graph with no
                # arguments, so the injected one is None on every real request and this
                # call raised AttributeError into the retry handler -- which read as a
                # model error rather than as no model at all.
                raw_response = client.invoke(current_prompt)
            except Exception as exc:
                exc_name = type(exc).__name__
                logger.error("LLM client invoke error: %s: %s", exc_name, exc)
                # Raise the correctly-typed retry exception so call_with_retry_typed
                # activates the right retry mode (backoff for rate limit, truncation
                # for timeout). Check by class name for client-agnostic matching.
                if "RateLimit" in exc_name or "rate_limit" in str(exc).lower():
                    raise RateLimitRetryError(str(exc)) from exc
                if "Timeout" in exc_name or "timeout" in str(exc).lower():
                    raise TimeoutRetryError(str(exc)) from exc
                raise ValidationRetryError(str(exc)) from exc

            # Parse and validate via Pydantic — raise ValidationRetryError on failure
            try:
                debrief = node._parse_response(raw_response)
            except (ValidationError, json.JSONDecodeError) as exc:
                logger.warning("LLM response parse failure: %s: %s", type(exc).__name__, exc)
                raise ValidationRetryError(str(exc)) from exc

            # Unpack to state primitives (DebriefOutput discarded after this)
            result = dict(state)
            result["summary"] = debrief.summary
            result["crm_fields"] = json.dumps(debrief.crm_fields, ensure_ascii=False)
            result["email_draft"] = debrief.email_draft
            return result

        return llm_fn

    def _parse_response(self, raw_response: str) -> DebriefOutput:
        """Parse and validate raw LLM response into DebriefOutput.

        Handles two response formats:
        1. Pure JSON string (ideal case)
        2. JSON embedded in markdown code block (```json ... ```)

        Raises:
            json.JSONDecodeError: If response cannot be parsed as JSON.
            pydantic.ValidationError: If parsed JSON does not match DebriefOutput schema.
        """
        text = raw_response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:] if lines[0].startswith("```") else lines
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        parsed = json.loads(text)
        return DebriefOutput(**parsed)
