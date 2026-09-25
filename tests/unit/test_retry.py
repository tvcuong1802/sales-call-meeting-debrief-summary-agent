"""
Tests for src/retry.py — LLM retry policy.

Design doc §9. Tests cover the three failure modes:
- ValidationError/JSONDecodeError: retry once, then LLM_SCHEMA_FAILURE
- RateLimitError: exponential backoff, then LLM_RATE_LIMIT
- TimeoutError: truncate transcript, retry once, then LLM_TIMEOUT

Uses the typed API (call_with_retry_typed + explicit exception classes)
for clean, deterministic test coverage.
"""

from __future__ import annotations

import json
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.retry import (
    call_with_retry_typed,
    ValidationRetryError,
    RateLimitRetryError,
    TimeoutRetryError,
    ERROR_LLM_SCHEMA_FAILURE,
    ERROR_LLM_RATE_LIMIT,
    ERROR_LLM_TIMEOUT,
    STRICT_PROMPT_SUFFIX,
)
from src.schemas.state import initial_state


def _make_state(**overrides) -> dict:
    state = initial_state("annotated transcript content", "sess-test")
    state.update(overrides)
    return state


def _no_sleep(monkeypatch):
    """Patch time.sleep to avoid actual delays in tests."""
    monkeypatch.setattr("src.retry.time.sleep", lambda s: None)


class TestRetrySchemaFailure:
    """Tests for ValidationError/JSONDecodeError retry mode."""

    def test_schema_failure_retries_once(self):
        """On ValidationRetryError, llm_fn is called a second time."""
        call_count = {"n": 0}

        def llm_fn(state):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ValidationRetryError("bad schema")
            # Second attempt succeeds
            return {**state, "summary": "success"}

        state = _make_state()
        result = call_with_retry_typed(llm_fn, state)
        assert call_count["n"] == 2
        assert result["summary"] == "success"
        assert result["error"] == ""

    def test_schema_failure_second_attempt_sets_error(self):
        """If both attempts fail with ValidationRetryError, sets LLM_SCHEMA_FAILURE."""

        def llm_fn(state):
            raise ValidationRetryError("schema always fails")

        state = _make_state()
        result = call_with_retry_typed(llm_fn, state)
        assert result["error"] != ""
        error = json.loads(result["error"])
        assert error["code"] == ERROR_LLM_SCHEMA_FAILURE
        assert error["attempt"] == 2

    def test_schema_failure_retry_gets_strict_prompt(self):
        """Retry attempt receives annotated_transcript with STRICT_PROMPT_SUFFIX appended."""
        second_call_transcript = {"value": None}

        def llm_fn(state):
            if state.get("annotated_transcript", "").endswith(STRICT_PROMPT_SUFFIX):
                second_call_transcript["value"] = state["annotated_transcript"]
                return {**state, "summary": "ok"}
            raise ValidationRetryError("first attempt fails")

        state = _make_state(annotated_transcript="original transcript")
        result = call_with_retry_typed(llm_fn, state)
        assert second_call_transcript["value"] is not None
        assert second_call_transcript["value"].endswith(STRICT_PROMPT_SUFFIX)

    def test_schema_failure_increments_retry_count(self):
        """retry_count is incremented on the retry attempt."""

        def llm_fn(state):
            if state.get("retry_count", 0) == 0:
                raise ValidationRetryError("first attempt")
            return {**state, "summary": "success"}

        state = _make_state(retry_count=0)
        result = call_with_retry_typed(llm_fn, state)
        assert result["retry_count"] >= 1


class TestRetryRateLimit:
    """Tests for RateLimitError retry mode."""

    def test_rate_limit_backoff(self, monkeypatch):
        """On RateLimitRetryError, exponential backoff is applied."""
        slept = []
        monkeypatch.setattr("src.retry.time.sleep", lambda s: slept.append(s))
        call_count = {"n": 0}

        def llm_fn(state):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise RateLimitRetryError("rate limited")
            return {**state, "summary": "success after backoff"}

        config = {"llm": {"retry_max": 3, "retry_backoff_base": 1.0}}
        state = _make_state()
        result = call_with_retry_typed(llm_fn, state, config=config)
        assert result["summary"] == "success after backoff"
        # Should have slept at least twice (two retries)
        assert len(slept) >= 2

    def test_rate_limit_exhausted_sets_error(self, monkeypatch):
        """After retry_max attempts, sets LLM_RATE_LIMIT."""
        monkeypatch.setattr("src.retry.time.sleep", lambda s: None)

        def llm_fn(state):
            raise RateLimitRetryError("always rate limited")

        config = {"llm": {"retry_max": 2, "retry_backoff_base": 0.0}}
        state = _make_state()
        result = call_with_retry_typed(llm_fn, state, config=config)
        assert result["error"] != ""
        error = json.loads(result["error"])
        assert error["code"] == ERROR_LLM_RATE_LIMIT

    def test_rate_limit_increments_retry_count(self, monkeypatch):
        """retry_count is incremented on each rate limit retry."""
        monkeypatch.setattr("src.retry.time.sleep", lambda s: None)

        def llm_fn(state):
            if state.get("retry_count", 0) < 2:
                raise RateLimitRetryError("rate limited")
            return {**state, "summary": "ok"}

        config = {"llm": {"retry_max": 3, "retry_backoff_base": 0.0}}
        state = _make_state(retry_count=0)
        result = call_with_retry_typed(llm_fn, state, config=config)
        assert result["retry_count"] >= 2


class TestRetryTimeout:
    """Tests for TimeoutError retry mode."""

    def test_timeout_truncates_transcript(self):
        """On TimeoutRetryError, retry gets annotated_transcript truncated to 50%."""
        second_call_length = {"value": None}

        def llm_fn(state):
            transcript = state.get("annotated_transcript", "")
            if second_call_length["value"] is None and transcript.endswith("XENDX"):
                # First call
                raise TimeoutRetryError("timeout")
            elif len(transcript) < 20:
                # This is the truncated retry
                second_call_length["value"] = len(transcript)
                return {**state, "summary": "ok after truncation"}
            raise TimeoutRetryError("timeout")

        original = "A" * 40 + "XENDX"
        state = _make_state(annotated_transcript=original)

        # Fresh approach: track calls
        calls = []

        def llm_fn2(state):
            calls.append(len(state.get("annotated_transcript", "")))
            if len(calls) == 1:
                raise TimeoutRetryError("first timeout")
            return {**state, "summary": "ok"}

        original = "A" * 100
        state = _make_state(annotated_transcript=original)
        result = call_with_retry_typed(llm_fn2, state)
        assert len(calls) == 2
        # Second call should have ~50% length
        assert calls[1] == 50  # 100 // 2

    def test_timeout_second_attempt_sets_error(self):
        """If both attempts time out, sets LLM_TIMEOUT."""

        def llm_fn(state):
            raise TimeoutRetryError("always times out")

        state = _make_state(annotated_transcript="some transcript")
        result = call_with_retry_typed(llm_fn, state)
        assert result["error"] != ""
        error = json.loads(result["error"])
        assert error["code"] == ERROR_LLM_TIMEOUT
        assert error["attempt"] == 2

    def test_timeout_increments_retry_count(self):
        """retry_count is incremented on the timeout retry."""

        def llm_fn(state):
            if state.get("retry_count", 0) == 0:
                raise TimeoutRetryError("first timeout")
            return {**state, "summary": "ok"}

        state = _make_state(retry_count=0)
        result = call_with_retry_typed(llm_fn, state)
        assert result["retry_count"] >= 1


class TestRetrySuccess:
    """Tests for successful (no-retry) paths."""

    def test_success_first_attempt_no_retry(self):
        """On first attempt success, returns immediately without retry."""
        call_count = {"n": 0}

        def llm_fn(state):
            call_count["n"] += 1
            return {**state, "summary": "immediate success"}

        state = _make_state()
        result = call_with_retry_typed(llm_fn, state)
        assert call_count["n"] == 1
        assert result["summary"] == "immediate success"
        assert result["error"] == ""
        assert result["retry_count"] == 0

    def test_success_does_not_increment_retry_count(self):
        """On first attempt success, retry_count stays 0."""

        def llm_fn(state):
            return {**state, "summary": "ok"}

        state = _make_state(retry_count=0)
        result = call_with_retry_typed(llm_fn, state)
        assert result["retry_count"] == 0

    def test_error_codes_are_correct_constants(self):
        """Error code constants match design doc §9.2."""
        assert ERROR_LLM_SCHEMA_FAILURE == "LLM_SCHEMA_FAILURE"
        assert ERROR_LLM_RATE_LIMIT == "LLM_RATE_LIMIT"
        assert ERROR_LLM_TIMEOUT == "LLM_TIMEOUT"


class TestTheRetryMatchersAreCatchable:
    """Every type in an except tuple must inherit BaseException.

    `openai.Timeout` is httpx's timeout CONFIG class and does not, so Python refused the
    except clause itself: "catching classes that do not inherit from BaseException is not
    allowed". Whenever the openai package was present -- which it is inside the image --
    the timeout retry path crashed instead of retrying. No test caught it because the
    suite runs without the package; mypy 2.1.0 in CI did.
    """

    def test_every_timeout_type_is_an_exception(self):
        from src.retry import _is_timeout_error

        for t in _is_timeout_error._exception_types():
            assert isinstance(t, type) and issubclass(t, BaseException), t

    def test_every_rate_limit_type_is_an_exception(self):
        from src.retry import _is_rate_limit_error

        for t in _is_rate_limit_error._exception_types():
            assert isinstance(t, type) and issubclass(t, BaseException), t

    def test_the_timeout_tuple_can_actually_be_caught(self):
        # The assertion above checks the types; this one checks the except clause, which
        # is where the TypeError was actually raised.
        from src.retry import _is_timeout_error

        try:
            raise TimeoutError("probe")
        except _is_timeout_error._exception_types():
            caught = True
        assert caught
