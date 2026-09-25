"""
Tests for src/nodes/nuance_classify.py — NuanceClassifyNode.

Design doc §5.2.
"""

from __future__ import annotations

import os
import sys
import tempfile

import yaml

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.nodes.nuance_classify import NuanceClassifyNode
from src.schemas.state import initial_state


def _make_state(**overrides) -> dict:
    state = initial_state("test transcript", "sess-test")
    state.update(overrides)
    return state


def _write_tmp_yaml(patterns: list) -> str:
    """Write a temporary nuance YAML file and return its path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.dump(patterns, tmp)
    tmp.close()
    return tmp.name


class TestNuanceClassifyNode:
    """Tests for NuanceClassifyNode."""

    def setup_method(self):
        self.node = NuanceClassifyNode()

    def _config_with_path(self, path: str) -> dict:
        return {"nuance": {"dictionary_path": path}}

    # ── Short-circuit ────────────────────────────────────────────────────────

    def test_short_circuit_on_error(self):
        """If state["error"] is non-empty, node returns state unchanged."""
        import json

        state = _make_state(
            cleaned_transcript="前向きに検討します",
            error=json.dumps({"code": "S1_INPUT_REJECTED", "attempt": 1}),
        )
        result = self.node.execute(state)
        # annotated_transcript must NOT be populated (short-circuit)
        assert result["annotated_transcript"] == ""
        assert result["error"] != ""

    # ── Pattern annotation ────────────────────────────────────────────────────

    def test_decline_pattern_annotated(self):
        """「前向きに検討します」 is annotated with [NUANCE:DECLINE]."""
        patterns = [{"pattern": "前向きに検討します", "tag": "DECLINE"}]
        yaml_path = _write_tmp_yaml(patterns)
        try:
            state = _make_state(cleaned_transcript="ご提案ありがとう。前向きに検討します。")
            result = NuanceClassifyNode(config=self._config_with_path(yaml_path)).execute(state)
            assert "[NUANCE:DECLINE]" in result["annotated_transcript"]
        finally:
            os.unlink(yaml_path)

    def test_defer_pattern_annotated(self):
        """「持ち帰って検討します」 is annotated with [NUANCE:DEFER]."""
        patterns = [{"pattern": "持ち帰って検討します", "tag": "DEFER"}]
        yaml_path = _write_tmp_yaml(patterns)
        try:
            state = _make_state(cleaned_transcript="持ち帰って検討します。また連絡します。")
            result = NuanceClassifyNode(config=self._config_with_path(yaml_path)).execute(state)
            assert "[NUANCE:DEFER]" in result["annotated_transcript"]
        finally:
            os.unlink(yaml_path)

    def test_soft_decline_pattern_annotated(self):
        """「ちょっと難しいかもしれません」 is annotated with [NUANCE:SOFT_DECLINE]."""
        patterns = [{"pattern": "ちょっと難しいかもしれません", "tag": "SOFT_DECLINE"}]
        yaml_path = _write_tmp_yaml(patterns)
        try:
            state = _make_state(cleaned_transcript="予算的にちょっと難しいかもしれません。")
            result = NuanceClassifyNode(config=self._config_with_path(yaml_path)).execute(state)
            assert "[NUANCE:SOFT_DECLINE]" in result["annotated_transcript"]
        finally:
            os.unlink(yaml_path)

    def test_positive_pattern_annotated(self):
        """「ぜひ」 is annotated with [NUANCE:POSITIVE]."""
        patterns = [{"pattern": "ぜひ", "tag": "POSITIVE"}]
        yaml_path = _write_tmp_yaml(patterns)
        try:
            state = _make_state(cleaned_transcript="ぜひ進めましょう。")
            result = NuanceClassifyNode(config=self._config_with_path(yaml_path)).execute(state)
            assert "[NUANCE:POSITIVE]" in result["annotated_transcript"]
        finally:
            os.unlink(yaml_path)

    # ── No annotation needed ──────────────────────────────────────────────────

    def test_no_annotation_needed(self):
        """Transcript with no nuance patterns returns unchanged text."""
        patterns = [{"pattern": "前向きに検討します", "tag": "DECLINE"}]
        yaml_path = _write_tmp_yaml(patterns)
        try:
            state = _make_state(cleaned_transcript="本日はありがとうございました。")
            result = NuanceClassifyNode(config=self._config_with_path(yaml_path)).execute(state)
            assert result["annotated_transcript"] == "本日はありがとうございました。"
            assert "[NUANCE:" not in result["annotated_transcript"]
        finally:
            os.unlink(yaml_path)

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_empty_transcript(self):
        """Empty string input produces annotated_transcript='' without crash."""
        patterns = [{"pattern": "前向きに検討します", "tag": "DECLINE"}]
        yaml_path = _write_tmp_yaml(patterns)
        try:
            state = _make_state(cleaned_transcript="")
            result = NuanceClassifyNode(config=self._config_with_path(yaml_path)).execute(state)
            assert result["annotated_transcript"] == ""
        finally:
            os.unlink(yaml_path)

    def test_missing_dictionary_file_graceful(self):
        """Missing dictionary file degrades gracefully — annotated_transcript = cleaned_transcript."""
        state = _make_state(cleaned_transcript="前向きに検討します。")
        result = NuanceClassifyNode(config=self._config_with_path("/nonexistent/nuance.yaml")).execute(state)
        # Graceful degradation: annotated_transcript = cleaned_transcript, no crash, no error
        assert result["annotated_transcript"] == "前向きに検討します。"
        assert result.get("error", "") == ""

    def test_multiple_patterns_all_annotated(self):
        """Multiple nuance patterns in same transcript all get annotated."""
        patterns = [
            {"pattern": "前向きに検討します", "tag": "DECLINE"},
            {"pattern": "ぜひ", "tag": "POSITIVE"},
        ]
        yaml_path = _write_tmp_yaml(patterns)
        try:
            state = _make_state(cleaned_transcript="ぜひ進めましょう。一方でコストは前向きに検討します。")
            result = NuanceClassifyNode(config=self._config_with_path(yaml_path)).execute(state)
            assert "[NUANCE:POSITIVE]" in result["annotated_transcript"]
            assert "[NUANCE:DECLINE]" in result["annotated_transcript"]
        finally:
            os.unlink(yaml_path)

    def test_annotation_appended_after_match(self):
        """Annotation tag is placed immediately after the matched phrase."""
        patterns = [{"pattern": "前向きに検討します", "tag": "DECLINE"}]
        yaml_path = _write_tmp_yaml(patterns)
        try:
            state = _make_state(cleaned_transcript="前向きに検討します。")
            result = NuanceClassifyNode(config=self._config_with_path(yaml_path)).execute(state)
            assert "前向きに検討します[NUANCE:DECLINE]" in result["annotated_transcript"]
        finally:
            os.unlink(yaml_path)

    def test_uses_bundled_nuance_ja_yaml(self):
        """Default path config/nuance_ja.yaml is loaded when no config provided."""
        # Run from repo root where config/nuance_ja.yaml exists
        original_dir = os.getcwd()
        try:
            os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
            state = _make_state(cleaned_transcript="前向きに検討します。")
            result = self.node.execute(state)  # no config → uses default path
            assert "[NUANCE:DECLINE]" in result["annotated_transcript"]
        finally:
            os.chdir(original_dir)

    def test_invalid_regex_pattern_skipped(self):
        """Invalid regex pattern in dictionary is skipped without crashing."""
        patterns = [
            {"pattern": "[invalid regex(", "tag": "DECLINE"},
            {"pattern": "ぜひ", "tag": "POSITIVE"},
        ]
        yaml_path = _write_tmp_yaml(patterns)
        try:
            state = _make_state(cleaned_transcript="ぜひ進めましょう。")
            result = NuanceClassifyNode(config=self._config_with_path(yaml_path)).execute(state)
            # Invalid pattern skipped, valid pattern still works
            assert "[NUANCE:POSITIVE]" in result["annotated_transcript"]
        finally:
            os.unlink(yaml_path)


class TestAnnotateTranscriptNoDoubleTags:
    """Regression tests for double-tagging fix (issue #21).

    Verifies that overlapping patterns in nuance_ja.yaml produce exactly
    one [NUANCE:TAG] per phrase — not two (old sequential re.sub behaviour).
    """

    def setup_method(self):
        self.node = NuanceClassifyNode()

    def _config_with_path(self, path: str) -> dict:
        return {"configurable": {"nuance": {"dictionary_path": path}}}

    def test_compound_defer_no_double_tag(self, tmp_path):
        """持ち帰って検討させていただきます → DEFER only, not DEFER+DECLINE."""
        patterns = [
            {"pattern": "持ち帰って検討させていただきます", "tag": "DEFER"},
            {"pattern": "検討させていただきます", "tag": "DECLINE"},
        ]
        yaml_path = str(tmp_path / "nuance.yaml")
        import yaml

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(patterns, f, allow_unicode=True)

        state = {"cleaned_transcript": "持ち帰って検討させていただきます", "error": ""}
        result = NuanceClassifyNode(config=self._config_with_path(yaml_path)).execute(state)
        annotated = result["annotated_transcript"]

        assert annotated.count("[NUANCE:") == 1, f"Expected exactly 1 tag, got: {repr(annotated)}"
        assert "[NUANCE:DEFER]" in annotated, f"Expected DEFER tag, got: {repr(annotated)}"
        assert "[NUANCE:DECLINE]" not in annotated, f"Unexpected DECLINE tag (double-tag regression): {repr(annotated)}"

    def test_compound_decline_no_double_tag(self, tmp_path):
        """前向きに検討させていただきます → DECLINE only, not DECLINE+DECLINE."""
        patterns = [
            {"pattern": "前向きに検討させていただきます", "tag": "DECLINE"},
            {"pattern": "検討させていただきます", "tag": "DECLINE"},
        ]
        yaml_path = str(tmp_path / "nuance.yaml")
        import yaml

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(patterns, f, allow_unicode=True)

        state = {"cleaned_transcript": "前向きに検討させていただきます", "error": ""}
        result = NuanceClassifyNode(config=self._config_with_path(yaml_path)).execute(state)
        annotated = result["annotated_transcript"]

        assert annotated.count("[NUANCE:") == 1, f"Expected exactly 1 DECLINE tag, got: {repr(annotated)}"

    def test_positive_prefix_no_double_tag(self, tmp_path):
        """ぜひ進めましょう → POSITIVE only, not two POSITIVE tags."""
        patterns = [
            {"pattern": "ぜひ進めましょう", "tag": "POSITIVE"},
            {"pattern": "ぜひ", "tag": "POSITIVE"},
        ]
        yaml_path = str(tmp_path / "nuance.yaml")
        import yaml

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(patterns, f, allow_unicode=True)

        state = {"cleaned_transcript": "ぜひ進めましょう", "error": ""}
        result = NuanceClassifyNode(config=self._config_with_path(yaml_path)).execute(state)
        annotated = result["annotated_transcript"]

        assert annotated.count("[NUANCE:") == 1, f"Expected exactly 1 POSITIVE tag, got: {repr(annotated)}"
        assert annotated == "ぜひ進めましょう[NUANCE:POSITIVE]", (
            f"Expected compound match tagged, got: {repr(annotated)}"
        )

    def test_bundled_yaml_no_double_tags_defer(self):
        """Integration: bundled nuance_ja.yaml produces no double tags for DEFER phrase."""
        import os

        original_dir = os.getcwd()
        try:
            os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
            state = {"cleaned_transcript": "持ち帰って検討させていただきます。", "error": ""}
            result = self.node.execute(state)
            annotated = result["annotated_transcript"]
            assert annotated.count("[NUANCE:") == 1, f"Bundled yaml double-tag regression: {repr(annotated)}"
            assert "[NUANCE:DEFER]" in annotated
            assert "[NUANCE:DECLINE]" not in annotated
        finally:
            os.chdir(original_dir)
