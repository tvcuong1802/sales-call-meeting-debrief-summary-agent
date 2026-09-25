"""
NuanceClassifyNode — CMN-C1-042 Sales Call & Meeting Debrief Summary Agent.

Pre-process slot node. Tags Japanese polite-rejection and stalling phrases in the
cleaned transcript before the LLM call, so the LLM can correctly classify deal
status rather than misreading politeness as agreement.

Design doc §5.2. No LLM call. Deterministic pattern-matching transform.
"""

from __future__ import annotations

import logging
import re
from typing import Any, ClassVar

import yaml

from framework.nodes.function_node import FunctionNode
from framework.schemas.invocation_context import TrustLevel
from src.utils.audit import emit_trace_event
from src.services.app_config import app_config

logger = logging.getLogger(__name__)


class NuanceClassifyNode(FunctionNode):
    """Tag Japanese business nuance phrases in the cleaned transcript.

    Scans cleaned_transcript for configured patterns and injects inline
    nuance annotations (e.g., [NUANCE:DECLINE], [NUANCE:DEFER]) so the
    downstream LLM call can correctly classify deal status.

    Processing:
    1. Short-circuit if state["error"] non-empty
    2. Load nuance dictionary from config path (injectable, not hardcoded)
    3. Scan cleaned_transcript for configured patterns, annotate inline
    4. Write annotated_transcript
    5. S-4 domain trace events: nuance_classified, nuance_classify_error

    Config key: nuance.dictionary_path -> path to YAML with pattern->tag mapping.
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
        """Execute nuance classification.

        Args:
            state: Current DebriefState dict.
            config: Optional framework config (carries configurable context).

        Returns:
            Updated state dict with annotated_transcript set.
        """
        try:
            # Short-circuit if upstream error
            if state.get("error"):
                return state

            cleaned = state.get("cleaned_transcript", "")

            # Load nuance dictionary from config path
            dictionary_path = self._get_dictionary_path(self._cfg())
            patterns = self._load_nuance_patterns(dictionary_path)

            # Annotate transcript with nuance tags
            annotated = self._annotate_transcript(cleaned, patterns)

            result = dict(state)
            result["annotated_transcript"] = annotated

            emit_trace_event(
                "nuance_classified",
                {
                    "status": "success",
                    "annotations_applied": annotated != cleaned,
                },
                state,
            )

            return result

        except Exception as exc:
            emit_trace_event(
                "nuance_classify_error",
                {
                    "node": self.__class__.__name__,
                    "error": type(exc).__name__,
                },
                state,
            )
            logger.error("NuanceClassifyNode error: %s: %s", type(exc).__name__, exc)
            # Non-fatal: return state with annotated_transcript = cleaned_transcript
            result = dict(state)
            if not result.get("annotated_transcript"):
                result["annotated_transcript"] = state.get("cleaned_transcript", "")
            return result

    def _get_dictionary_path(self, config: dict[str, Any] | None) -> str:
        """Extract nuance dictionary path from config.

        Args:
            config: Framework config dict. Expected structure:
                    config["configurable"]["nuance"]["dictionary_path"] or
                    config["nuance"]["dictionary_path"].

        Returns:
            Path string to the YAML nuance dictionary.
        """
        default_path = "config/nuance_ja.yaml"

        if config is None:
            return default_path

        # Try framework configurable path first
        configurable = config.get("configurable", {})
        nuance_cfg = configurable.get("nuance", config.get("nuance", {}))
        dictionary_path: str = nuance_cfg.get("dictionary_path", default_path)
        return dictionary_path

    def _load_nuance_patterns(self, dictionary_path: str) -> list[dict[str, str]]:
        """Load nuance pattern list from YAML file.

        Args:
            dictionary_path: Path to YAML file with entries:
                - pattern: <regex string>
                  tag: <DECLINE|DEFER|SOFT_DECLINE|POSITIVE>

        Returns:
            List of {"pattern": str, "tag": str} dicts ordered for matching.
            Returns empty list if file not found (graceful degradation).
        """
        try:
            with open(dictionary_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, list):
                logger.warning("Nuance dictionary %s is not a list — skipping", dictionary_path)
                return []
            return data
        except FileNotFoundError:
            logger.warning("Nuance dictionary not found: %s — skipping annotation", dictionary_path)
            return []

    def _annotate_transcript(self, transcript: str, patterns: list[dict[str, str]]) -> str:
        """Scan transcript and inject inline nuance annotations — single-pass.

        Uses a combined regex (all patterns joined by |, longest/most-specific first)
        so a compound phrase like 「持ち帰って検討させていただきます」 is tagged exactly once as DEFER
        and never as DEFER+DECLINE (which sequential re.sub produces when
        「検討させていただきます」 re-matches the already-annotated string).

        Pattern ordering in the YAML determines priority: the first matching group
        in the alternation wins. Longest/most-specific patterns must appear first
        in nuance_ja.yaml.

        Args:
            transcript: Cleaned transcript text.
            patterns: List of {"pattern": regex_str, "tag": tag_str} dicts,
                      ordered most-specific first.

        Returns:
            Annotated transcript string (or original if no patterns matched).
        """
        if not transcript or not patterns:
            return transcript

        # Build combined regex with named groups — single pass prevents double-tagging
        tag_map: dict[str, str] = {}
        combined_parts: list[str] = []
        for i, entry in enumerate(patterns):
            pattern = entry.get("pattern", "")
            tag = entry.get("tag", "")
            if not pattern or not tag:
                continue
            group_name = f"g{i}"
            try:
                re.compile(pattern)  # validate individual pattern first
            except re.error as exc:
                logger.warning("Invalid nuance pattern %r: %s — skipping", pattern, exc)
                continue
            combined_parts.append(f"(?P<{group_name}>{pattern})")
            tag_map[group_name] = tag

        if not combined_parts:
            return transcript

        combined = "|".join(combined_parts)
        try:
            compiled = re.compile(combined)
        except re.error as exc:
            logger.warning("Combined nuance regex compile failed: %s — skipping annotation", exc)
            return transcript

        def _replacer(m: re.Match[str]) -> str:
            for name, tag in tag_map.items():
                if m.group(name) is not None:
                    return m.group(0) + f"[NUANCE:{tag}]"
            return m.group(0)  # unreachable but safe fallback

        return compiled.sub(_replacer, transcript)
