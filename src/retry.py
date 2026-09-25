"""
LLM retry policy for CMN-C1-042 — Sales Call & Meeting Debrief Summary Agent.

Standalone utility module used by StructuredExtractNode (issue #11).
Implements the three failure mode retry policy per design doc §9.

This module is NOT a node. Import and call call_with_retry() from StructuredExtractNode.

Design doc §9.
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Error codes (design doc §9.2)
ERROR_LLM_SCHEMA_FAILURE = "LLM_SCHEMA_FAILURE"
ERROR_LLM_RATE_LIMIT = "LLM_RATE_LIMIT"
ERROR_LLM_TIMEOUT = "LLM_TIMEOUT"

# Default retry config (overridden by config["llm"])
DEFAULT_RETRY_MAX = 3
DEFAULT_BACKOFF_BASE = 1.0
DEFAULT_TIMEOUT_SECONDS = 30

# Stricter prompt suffix injected on schema validation retry (design doc §9.3)
STRICT_PROMPT_SUFFIX = (
    "\n\n[STRICT MODE] Your previous response did not conform to the required JSON schema. "
    "You MUST respond with valid JSON that exactly matches the DebriefOutput schema: "
    '{"summary": str, "crm_fields": dict, "email_draft": str}. '
    "Do not include any text outside the JSON object."
)


class LLMRetryPolicy:
    """Retry policy for LLM calls in StructuredExtractNode.

    Handles three failure modes per design doc §9.3:
    - ValidationError/JSONDecodeError: retry once with stricter prompt
    - RateLimitError (HTTP 429): exponential backoff with jitter
    - TimeoutError: retry once with 50% transcript truncation

    Usage:
        policy = LLMRetryPolicy(config)
        result_state = policy.call(llm_fn, state)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize retry policy from config.

        Args:
            config: Agent config dict. Reads config["llm"]["retry_max"],
                    config["llm"]["retry_backoff_base"].

        Note:
            Timeout enforcement is delegated entirely to the LLM client.
            When the client raises TimeoutError (or TimeoutRetryError), the
            active execution path (call_with_retry_typed) handles the retry.
            There is no separate `timeout_seconds` config key — set the
            timeout on the LLM client itself via `llm.timeout` in config.yaml.
        """
        llm_cfg = (config or {}).get("llm", {})
        self.retry_max: int = llm_cfg.get("retry_max", DEFAULT_RETRY_MAX)
        self.backoff_base: float = llm_cfg.get("retry_backoff_base", DEFAULT_BACKOFF_BASE)

    def call(
        self,
        llm_fn: Callable[[dict[str, Any]], dict[str, Any]],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Call llm_fn with retry policy applied.

        Args:
            llm_fn: Callable that takes state and returns updated state dict.
                    May raise ValidationError, JSONDecodeError, RateLimitError, TimeoutError.
            state: Current DebriefState dict.

        Returns:
            Updated state dict. On exhaustion, state["error"] is set with
            structured error code. Never raises.
        """
        return call_with_retry(
            llm_fn,
            state,
            config={
                "llm": {
                    "retry_max": self.retry_max,
                    "retry_backoff_base": self.backoff_base,
                }
            },
        )


def call_with_retry(
    llm_fn: Callable[[dict[str, Any]], dict[str, Any]],
    state: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call llm_fn with the three-mode retry policy.

    Three failure modes (design doc §9.3):

    a. ValidationError/JSONDecodeError:
       - Retry once with stricter prompt suffix appended to annotated_transcript
       - On second failure: set state["error"] = LLM_SCHEMA_FAILURE

    b. RateLimitError (HTTP 429 or RateLimitError exception):
       - Exponential backoff with jitter, up to config["llm"]["retry_max"] attempts
       - On exhaustion: set state["error"] = LLM_RATE_LIMIT

    c. TimeoutError:
       - Retry once with annotated_transcript tail-truncated to 50%
       - On second timeout: set state["error"] = LLM_TIMEOUT

    In all cases: state["retry_count"] incremented on each retry.
    S-4 trace event "llm_retry" emitted on each retry.

    Args:
        llm_fn: Callable that takes state dict and returns updated state dict.
        state: Current DebriefState dict.
        config: Config dict with llm.retry_max, llm.retry_backoff_base.

    Returns:
        Updated state dict. On failure, state["error"] is set. Never raises.
    """
    llm_cfg = (config or {}).get("llm", {})
    retry_max: int = llm_cfg.get("retry_max", DEFAULT_RETRY_MAX)
    backoff_base: float = llm_cfg.get("retry_backoff_base", DEFAULT_BACKOFF_BASE)

    current_state = dict(state)

    # ── Attempt 1 ────────────────────────────────────────────────────────────
    try:
        return llm_fn(current_state)

    except _is_validation_error() as exc:  # type: ignore[misc]
        logger.warning("LLM schema failure on attempt 1: %s", type(exc).__name__)
        current_state = _emit_retry_trace(current_state, attempt=1, reason="schema_validation_failure")
        current_state["retry_count"] = current_state.get("retry_count", 0) + 1

        # Retry with stricter prompt (append suffix to annotated_transcript)
        retry_state = dict(current_state)
        retry_state["annotated_transcript"] = retry_state.get("annotated_transcript", "") + STRICT_PROMPT_SUFFIX
        try:
            return llm_fn(retry_state)
        except Exception:
            return _set_error(current_state, ERROR_LLM_SCHEMA_FAILURE, attempt=2)

    except _is_rate_limit_error() as exc:  # type: ignore[misc]
        logger.warning("LLM rate limit on attempt 1: %s", type(exc).__name__)
        return _handle_rate_limit(llm_fn, current_state, attempt=1, retry_max=retry_max, backoff_base=backoff_base)

    except _is_timeout_error() as exc:  # type: ignore[misc]
        logger.warning("LLM timeout on attempt 1: %s", type(exc).__name__)
        current_state = _emit_retry_trace(current_state, attempt=1, reason="timeout")
        current_state["retry_count"] = current_state.get("retry_count", 0) + 1

        # Retry with 50% tail-truncated transcript
        retry_state = dict(current_state)
        transcript = retry_state.get("annotated_transcript", "")
        retry_state["annotated_transcript"] = transcript[: len(transcript) // 2]
        try:
            return llm_fn(retry_state)
        except Exception:
            return _set_error(current_state, ERROR_LLM_TIMEOUT, attempt=2)

    except Exception as exc:
        logger.error("LLM unexpected error on attempt 1: %s: %s", type(exc).__name__, exc)
        return _set_error(current_state, ERROR_LLM_SCHEMA_FAILURE, attempt=1)


# ── Failure mode helpers ──────────────────────────────────────────────────────


def _handle_rate_limit(
    llm_fn: Callable[[dict[str, Any]], dict[str, Any]],
    state: dict[str, Any],
    attempt: int,
    retry_max: int,
    backoff_base: float,
) -> dict[str, Any]:
    """Exponential backoff retry for rate limit errors."""
    current_state = dict(state)

    for n in range(1, retry_max + 1):
        sleep_time = backoff_base * (2 ** (n - 1)) + random.uniform(0.0, 0.5)
        logger.info("Rate limit backoff: attempt %d/%d, sleeping %.2fs", n, retry_max, sleep_time)
        time.sleep(sleep_time)
        current_state = _emit_retry_trace(current_state, attempt=n, reason="rate_limit")
        current_state["retry_count"] = current_state.get("retry_count", 0) + 1

        try:
            return llm_fn(current_state)
        except _is_rate_limit_error():  # type: ignore[misc]
            continue
        except Exception as exc:
            logger.error("LLM non-rate-limit error during backoff: %s", type(exc).__name__)
            return _set_error(current_state, ERROR_LLM_RATE_LIMIT, attempt=n + 1)

    return _set_error(current_state, ERROR_LLM_RATE_LIMIT, attempt=retry_max + 1)


def _set_error(state: dict[str, Any], code: str, attempt: int) -> dict[str, Any]:
    """Set state["error"] with structured error JSON."""
    result = dict(state)
    result["error"] = json.dumps(
        {
            "code": code,
            "attempt": attempt,
            "node": "StructuredExtractNode",
        }
    )
    logger.error("LLM error set: code=%s attempt=%d", code, attempt)
    return result


def _emit_retry_trace(state: dict[str, Any], attempt: int, reason: str) -> dict[str, Any]:
    """Emit S-4 trace event for LLM retry. Returns updated state."""
    # In production, this would call self._emit_trace_event() on the node.
    # As a standalone utility, we log the event.
    logger.info(
        "S-4 llm_retry: attempt=%d reason=%s retry_count=%d",
        attempt,
        reason,
        state.get("retry_count", 0),
    )
    return state


# ── Exception type matchers (compatible with stub + production frameworks) ────


class _is_validation_error:
    """Context manager / base class for validation error matching."""

    _types = (ValueError, TypeError)

    def __class_getitem__(cls, item: Any) -> type[_is_validation_error]:
        return cls

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

    @classmethod
    def _exception_types(cls) -> tuple[type[BaseException], ...]:
        try:
            from pydantic import ValidationError as PydanticValidationError

            return (PydanticValidationError, json.JSONDecodeError, ValueError)
        except ImportError:
            return (json.JSONDecodeError, ValueError)


class _is_rate_limit_error:
    """Matches rate limit / HTTP 429 errors."""

    @classmethod
    def _exception_types(cls) -> tuple[type[BaseException], ...]:
        types: list[type[BaseException]] = []
        # Optional: if the openai SDK is installed, use its typed RateLimitError for
        # precise retry routing. Falls back safely to the generic Exception path if
        # openai is absent or at an incompatible version (AttributeError swallowed).
        # openai is NOT a required production dependency — do not add to pyproject.toml
        # unless the deployment explicitly depends on it.
        try:
            import openai

            types.append(openai.RateLimitError)
        except (ImportError, AttributeError):
            pass
        # Generic fallback: any exception whose name contains RateLimit
        return tuple(types) if types else (Exception,)


class _is_timeout_error:
    """Matches timeout errors."""

    @classmethod
    def _exception_types(cls) -> tuple[type[BaseException], ...]:
        types: list[type[BaseException]] = [TimeoutError]
        # `openai.APITimeoutError`, NOT `openai.Timeout`. The latter is httpx's timeout
        # CONFIG class -- it does not inherit BaseException, and Python refuses to catch
        # it: `TypeError: catching classes that do not inherit from BaseException is not
        # allowed`, raised from the except clause itself. So whenever the openai package
        # was installed -- which it is inside the image -- the timeout retry path did not
        # retry, it crashed. mypy 2.1.0 in CI is what surfaced it; mypy 1.10.0 did not,
        # and no test covered it because the tests run without the package.
        #
        # Same pattern as _is_rate_limit_error — optional, non-required dependency.
        try:
            import openai

            types.append(openai.APITimeoutError)
        except (ImportError, AttributeError):
            pass
        return tuple(types)


# ── Public exception-based API (for direct use in StructuredExtractNode) ─────


class ValidationRetryError(Exception):
    """Raised by StructuredExtractNode to trigger schema validation retry."""

    pass


class RateLimitRetryError(Exception):
    """Raised by StructuredExtractNode to trigger rate limit retry."""

    pass


class TimeoutRetryError(Exception):
    """Raised by StructuredExtractNode to trigger timeout retry."""

    pass


def call_with_retry_typed(
    llm_fn: Callable[[dict[str, Any]], dict[str, Any]],
    state: dict[str, Any],
    config: dict[str, Any] | None = None,
    on_retry: Optional[Callable[[int, str, str], None]] = None,
) -> dict[str, Any]:
    """Typed version of call_with_retry using explicit retry exception classes.

    StructuredExtractNode should raise ValidationRetryError, RateLimitRetryError,
    or TimeoutRetryError to trigger the appropriate retry mode.

    Args:
        llm_fn: LLM callable. Raises typed retry exceptions on failure.
        state: Current DebriefState dict.
        config: Config dict. Reads llm.retry.max_attempts and llm.retry.backoff_factor
                (config.yaml format). Falls back to llm.retry_max / llm.retry_backoff_base
                for backward compatibility.
        on_retry: Optional callback invoked on each retry attempt.
                  Signature: on_retry(attempt: int, error_type: str, node_name: str) -> None.
                  Used by StructuredExtractNode to emit node_retry trace events.

    Returns:
        Updated state dict. On failure, state["error"] is set. Never raises.
    """
    llm_cfg = (config or {}).get("llm", {})
    # Support both config.yaml format (retry.max_attempts) and legacy flat format (retry_max)
    retry_nested = llm_cfg.get("retry", {})
    retry_max = retry_nested.get("max_attempts", llm_cfg.get("retry_max", DEFAULT_RETRY_MAX))
    backoff_base = retry_nested.get("backoff_factor", llm_cfg.get("retry_backoff_base", DEFAULT_BACKOFF_BASE))

    current_state = dict(state)

    try:
        return llm_fn(current_state)

    except ValidationRetryError:
        logger.warning("LLM schema failure on attempt 1 (typed)")
        if on_retry is not None:
            on_retry(1, "LLM_SCHEMA_FAILURE", "StructuredExtractNode")
        current_state = _emit_retry_trace(current_state, attempt=1, reason="schema_validation_failure")
        current_state["retry_count"] = current_state.get("retry_count", 0) + 1
        retry_state = dict(current_state)
        retry_state["annotated_transcript"] = retry_state.get("annotated_transcript", "") + STRICT_PROMPT_SUFFIX
        try:
            return llm_fn(retry_state)
        except Exception:
            return _set_error(current_state, ERROR_LLM_SCHEMA_FAILURE, attempt=2)

    except RateLimitRetryError:
        logger.warning("LLM rate limit on attempt 1 (typed)")
        return _handle_rate_limit_typed(
            llm_fn, current_state, retry_max=retry_max, backoff_base=backoff_base, on_retry=on_retry
        )

    except TimeoutRetryError:
        logger.warning("LLM timeout on attempt 1 (typed)")
        if on_retry is not None:
            on_retry(1, "LLM_TIMEOUT", "StructuredExtractNode")
        current_state = _emit_retry_trace(current_state, attempt=1, reason="timeout")
        current_state["retry_count"] = current_state.get("retry_count", 0) + 1
        retry_state = dict(current_state)
        transcript = retry_state.get("annotated_transcript", "")
        retry_state["annotated_transcript"] = transcript[: len(transcript) // 2]
        try:
            return llm_fn(retry_state)
        except Exception:
            return _set_error(current_state, ERROR_LLM_TIMEOUT, attempt=2)

    except Exception as exc:
        logger.error("LLM unexpected error: %s: %s", type(exc).__name__, exc)
        return _set_error(current_state, ERROR_LLM_SCHEMA_FAILURE, attempt=1)


def _handle_rate_limit_typed(
    llm_fn: Callable[[dict[str, Any]], dict[str, Any]],
    state: dict[str, Any],
    retry_max: int,
    backoff_base: float,
    on_retry: Optional[Callable[[int, str, str], None]] = None,
) -> dict[str, Any]:
    """Rate limit retry loop (typed exceptions)."""
    current_state = dict(state)
    for n in range(1, retry_max + 1):
        sleep_time = backoff_base * (2 ** (n - 1)) + random.uniform(0.0, 0.5)
        logger.info("Rate limit backoff: attempt %d/%d, sleep %.2fs", n, retry_max, sleep_time)
        time.sleep(sleep_time)
        if on_retry is not None:
            on_retry(n, "LLM_RATE_LIMIT", "StructuredExtractNode")
        current_state = _emit_retry_trace(current_state, attempt=n, reason="rate_limit")
        current_state["retry_count"] = current_state.get("retry_count", 0) + 1
        try:
            return llm_fn(current_state)
        except RateLimitRetryError:
            continue
        except Exception as exc:
            logger.error("Non-rate-limit error during backoff: %s", type(exc).__name__)
            return _set_error(current_state, ERROR_LLM_RATE_LIMIT, attempt=n + 1)
    return _set_error(current_state, ERROR_LLM_RATE_LIMIT, attempt=retry_max + 1)
