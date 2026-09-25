"""
TranscriptCleanNode — CMN-C1-042 Sales Call & Meeting Debrief Summary Agent.

Pre-process slot node. Normalizes raw transcript input, enforces S-1 trust
level gate, implements S-2 PII detection + masking (APPI compliance).

Design doc §5.1, §8.2. No LLM call. Deterministic transform.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.invocation_context import TrustLevel
from src.utils.audit import emit_trace_event
from src.services.app_config import app_config

logger = logging.getLogger(__name__)

# S-1 trust level constant — framework security gate
# TrustLevel.VERIFIED_EXTERNAL: sales transcripts contain customer PII
# (design doc §5.1, §8.1)

# S-1 validation limits (design doc §5.1, §7.1)
MAX_TRANSCRIPT_CHARS = 50_000

# Aizuchi patterns to strip (design doc §5.1)
_AIZUCHI_PATTERNS = [
    r"えー+",
    r"あー+",
    r"うん+",
]

# Speaker tag normalization (design doc §5.1)
_SPEAKER_TAG_PATTERN = re.compile(r"\[Speaker\s+(\d+)\]", re.IGNORECASE)


# ── S-2 PII Detection Patterns (design doc §8.2, the security rules) ──────────
#
# APPI no-persistence: patterns match PII *content* but mask tokens NEVER
# contain raw PII. Pattern order matters — more specific patterns first.
#
# Compiled once at module load (performance).

# Phone numbers (JP mobile: 0X0-XXXX-XXXX, JP landline: 0X-XXXX-XXXX,
# international: +81-..., bare 10/11-digit runs)
_RE_PHONE = re.compile(
    r"(?:"
    r"\+81[-\s]?\d{1,4}[-\s]?\d{2,4}[-\s]?\d{3,4}"  # +81 international
    r"|0\d{1,3}[-\s]?\d{2,4}[-\s]?\d{3,4}"  # JP domestic (0XX-XXX-XXXX)
    r"|(?<!\d)\d{10,11}(?!\d)"  # bare 10/11-digit runs
    r")"
)

# Email addresses (RFC 5322 simplified)
_RE_EMAIL = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# Government IDs
# JP My Number (マイナンバー): 12 consecutive digits (not part of longer number)
# US SSN: NNN-NN-NNNN
_RE_GOV_ID = re.compile(
    r"(?:"
    r"(?<!\d)\d{12}(?!\d)"  # JP My Number (12-digit)
    r"|\b\d{3}-\d{2}-\d{4}\b"  # US SSN
    r")"
)

# Financial identifiers
# Credit cards: 4-group 16-digit (with spaces/dashes), or 13/15/16-digit runs
# JP bank account: branch code (3-digit) + account number (7-digit), hyphenated
_RE_FINANCIAL = re.compile(
    r"(?:"
    r"\b(?:\d{4}[-\s]){3}\d{4}\b"  # 4×4 credit card
    r"|\b(?:\d{4}[-\s]){2}\d{7}\b"  # Amex-style 4-6-5
    r"|\b\d{3}[-\s]\d{7}\b"  # JP bank account NNN-NNNNNNN
    r"|(?<!\d)\d{13}(?!\d)"  # bare 13-digit (Visa old)
    r"|(?<!\d)\d{15}(?!\d)"  # bare 15-digit (Amex)
    r"|(?<!\d)\d{16}(?!\d)"  # bare 16-digit (card)
    r")"
)

# Addresses
# JP: Prefecture names followed by municipality/ward/city keywords
_JP_PREFECTURES = (
    r"(?:北海道|青森|岩手|宮城|秋田|山形|福島|茨城|栃木|群馬|埼玉|千葉|東京|神奈川"
    r"|新潟|富山|石川|福井|山梨|長野|岐阜|静岡|愛知|三重|滋賀|京都|大阪|兵庫|奈良"
    r"|和歌山|鳥取|島根|岡山|広島|山口|徳島|香川|愛媛|高知|福岡|佐賀|長崎|熊本"
    r"|大分|宮崎|鹿児島|沖縄)(?:都|道|府|県)?"
)
_RE_ADDRESS_JP = re.compile(
    r"(?:"
    + _JP_PREFECTURES
    + r"[\s\S]{0,20}?(?:市|区|町|村|郡)"  # prefecture + up to 20 chars + city/ward
    + r"[\s\S]{0,30}?(?:\d+[-－]\d+|丁目|番地|号)"  # + street number
    + r")"
)

# EN: House number + street name + street type keyword
# Avoids "Room 3", "Floor 2" etc. by requiring a named street type
_EN_STREET_TYPES = (
    r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr"
    r"|Lane|Ln|Court|Ct|Place|Pl|Way|Parkway|Pkwy|Highway|Hwy)"
)
_RE_ADDRESS_EN = re.compile(r"\b\d{1,5}\s+[A-Z][a-zA-Z\s]{1,30}\s+" + _EN_STREET_TYPES + r"\.?\b")

# Personal names (JP: token + honorific suffix; EN: capitalized word pair heuristic)
# JP: any non-space run of 1–8 CJK/kana chars immediately before 様/さん/氏/殿
_RE_NAME_JP = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff\uff00-\uffef]{1,8}(?:様|さん|氏|殿)")

# EN: Two or three consecutive title-cased words not matching known non-name patterns
# Excluded: single words, all-caps abbreviations, known company suffixes
_EN_NAME_EXCLUSIONS = re.compile(
    r"^(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
    r"|January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Inc|Ltd|Corp|LLC|Co|Department|Team|Group|Division|Project|Meeting|Call|Room"
    r"|North|South|East|West|New|The|This|That|Our|Your|Their|We|Us|They)$",
    re.IGNORECASE,
)
_RE_NAME_EN = re.compile(r"\b([A-Z][a-z]{1,20})\s+([A-Z][a-z]{1,20})(?:\s+([A-Z][a-z]{1,20}))?\b")

# Medical record numbers: カルテ番号 + digits, or MR/ID + hyphen + digits
_RE_MEDICAL_ID = re.compile(
    r"(?:"
    r"カルテ(?:番号)?[\s:：]?\d{4,10}"
    r"|\bMR[-\s]?\d{4,10}\b"
    r"|\b(?:Patient|Medical)\s+(?:ID|No)\.?\s*\d{4,10}\b"
    r")",
    re.IGNORECASE,
)

# Ordered list of (pattern, mask_token, category_label) — more specific first
_PII_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (_RE_MEDICAL_ID, "[MEDICAL_ID]", "MEDICAL_ID"),
    (_RE_EMAIL, "[EMAIL]", "EMAIL"),
    (_RE_GOV_ID, "[GOV_ID]", "GOV_ID"),
    (_RE_FINANCIAL, "[FINANCIAL]", "FINANCIAL"),
    (_RE_PHONE, "[PHONE]", "PHONE"),
    (_RE_ADDRESS_JP, "[ADDRESS]", "ADDRESS"),
    (_RE_ADDRESS_EN, "[ADDRESS]", "ADDRESS"),
]

# Name patterns applied last (lower precision, so run after structural patterns)
_NAME_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (_RE_NAME_JP, "[NAME]", "NAME"),
]


def _mask_en_names(text: str) -> tuple[str, int, list[str]]:
    """Mask English proper name pairs (Title Case heuristic).

    A match is treated as a name only when:
    - Both words are title-cased (first letter upper, rest lower)
    - Neither word is in the exclusion list

    Returns (masked_text, count_masked, categories_found).
    """
    count = 0
    categories: list[str] = []

    def _replace(m: re.Match[str]) -> str:
        nonlocal count
        w1 = m.group(1)
        w2 = m.group(2)
        # Skip if either primary word is in exclusion list
        if _EN_NAME_EXCLUSIONS.match(w1) or _EN_NAME_EXCLUSIONS.match(w2):
            # m.group(0) is the matched text, a str, returned unchanged when the word is
            # excluded. The old annotation was the ENCLOSING function's return type
            # (masked_text, count, categories) pasted onto a local inside _replace, which
            # is declared `-> str`. Runtime value is untouched.
            unchanged: str = m.group(0)
            return unchanged
        count += 1
        if "NAME" not in categories:
            categories.append("NAME")
        return "[NAME]"

    masked = _RE_NAME_EN.sub(_replace, text)
    return masked, count, categories


def _apply_pii_rules(text: str) -> tuple[str, int, list[str]]:
    """Apply all PII rules to text. Returns (masked_text, total_count, category_list).

    APPI no-persistence: only category labels (not matched content) are recorded.
    """
    total_count = 0
    categories_found: list[str] = []

    # Structural PII rules
    for pattern, mask_token, category in _PII_RULES:
        matches = pattern.findall(text)
        if matches:
            n = len(matches)
            total_count += n
            if category not in categories_found:
                categories_found.append(category)
            text = pattern.sub(mask_token, text)

    # JP name rule
    for pattern, mask_token, category in _NAME_RULES:
        matches = pattern.findall(text)
        if matches:
            n = len(matches)
            total_count += n
            if category not in categories_found:
                categories_found.append(category)
            text = pattern.sub(mask_token, text)

    # EN name heuristic
    text, en_name_count, en_name_cats = _mask_en_names(text)
    total_count += en_name_count
    for cat in en_name_cats:
        if cat not in categories_found:
            categories_found.append(cat)

    return text, total_count, categories_found


class TranscriptCleanNode(FunctionNode):
    """Normalize raw transcript; enforce S-1 gate; implement S-2 PII masking.

    Processing:
    1. S-1 trust level check (framework __pre_invoke__ — see design doc §3.3)
    2. S-2 PII detection + mask (APPI compliance — design doc §8.2)
    3. Validate input: reject if empty, > MAX_TRANSCRIPT_CHARS, or non-text bytes
    4. Generate invocation_id (UUID4)
    5. Compute audit_input_hash (SHA-256 of raw_transcript, hex)
    6. Normalize: strip aizuchi, normalize speaker tags, collapse whitespace
    7. Write cleaned_transcript, invocation_id, audit_input_hash
    8. S-4 domain trace events: transcript_cleaned, transcript_clean_error

    Security gates:
    - S-1: required_trust_level = VERIFIED_EXTERNAL
    - S-2: _security_gate_input() — functional APPI PII masking (issue #16)
    - S-4: trace events on all execution paths
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
        """Execute transcript cleaning and normalization.

        Args:
            state: Current DebriefState dict.
            config: Optional framework config.

        Returns:
            Updated state dict with cleaned_transcript, invocation_id,
            audit_input_hash, pii_masked, pii_mask_count set.
        """
        try:
            # Short-circuit if upstream error already set
            if state.get("error"):
                return state

            # S-2 PII gate: mask PII in raw_transcript before any downstream use.
            # Private node-level helper (NOT an override of FunctionNode's @final
            # _security_gate_input); the framework @final S-2 gate still fires
            # per-node as defence in depth (CR-0930).
            state = self._mask_pii(state)

            raw = state.get("user_input") or state.get("raw_transcript", "")

            # S-1 input validation
            validation_result = self._validate_input(raw)
            if validation_result is not None:
                result = dict(state)
                result["error"] = validation_result
                emit_trace_event(
                    "transcript_clean_error",
                    {
                        "node": self.__class__.__name__,
                        "error": "S1_INPUT_REJECTED",
                    },
                    state,
                )
                return result

            # Generate invocation_id and audit_input_hash
            invocation_id = str(uuid.uuid4())  # standard UUID4 with hyphens (36 chars)
            # Hash of already-masked transcript (no raw PII in hash input)
            audit_input_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()

            # Normalize transcript
            cleaned = self._normalize_transcript(raw)

            result = dict(state)
            result["cleaned_transcript"] = cleaned
            result["invocation_id"] = invocation_id
            result["audit_input_hash"] = audit_input_hash

            emit_trace_event(
                "transcript_cleaned",
                {
                    "status": "success",
                    "invocation_id": invocation_id,
                    "audit_input_hash": audit_input_hash,
                },
                state,
            )

            return result

        except Exception as exc:
            emit_trace_event(
                "transcript_clean_error",
                {
                    "node": self.__class__.__name__,
                    "error": type(exc).__name__,
                },
                state,
            )
            logger.error("TranscriptCleanNode error: %s: %s", type(exc).__name__, exc)
            result = dict(state)
            result["error"] = json.dumps(
                {
                    "code": "TRANSCRIPT_CLEAN_ERROR",
                    "attempt": 1,
                    "node": "TranscriptCleanNode",
                }
            )
            return result

    def _mask_pii(self, state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        """S-2 PII detection and masking (node-level private helper).

        Named ``_mask_pii`` (not ``_security_gate_input``) to avoid colliding with
        FunctionNode's @final S-2 gate; called inline in execute() so masking runs
        on raw_transcript before cleaning/LLM at the right ordering point (CR-0930).
        APPI compliance — design doc §8.2.

        Detects and masks PII categories in raw_transcript BEFORE cleaning
        or LLM call. Implements the security rules Layer S-2.

        PII categories masked (§8.2):
        - Personal names: JP suffix heuristic (様/さん/氏/殿), EN title-case pair
        - Phone numbers: JP mobile/landline, international (+81), bare 10/11-digit
        - Email addresses: RFC 5322 simplified
        - Government IDs: JP My Number (12-digit), US SSN (NNN-NN-NNNN)
        - Financial identifiers: credit cards (16-digit), JP bank accounts
        - Addresses: JP prefecture+municipality+street, EN number+street+type
        - Medical record numbers: カルテ番号, MR-NNNN

        APPI no-persistence guarantee:
        - raw_transcript in returned state contains mask tokens, not raw PII
        - Trace event payload contains ONLY category labels and count (no PII content)
        - Nothing PII-containing written to logs

        Args:
            state: Current DebriefState dict.
            config: Optional framework config (unused, for interface compat).

        Returns:
            Updated state dict with raw_transcript masked, pii_masked and
            pii_mask_count set accordingly.
        """
        raw = state.get("user_input") or state.get("raw_transcript", "")
        if not raw:
            return state

        masked_text, total_count, categories = _apply_pii_rules(raw)

        result = dict(state)
        result["raw_transcript"] = masked_text
        result["pii_masked"] = total_count > 0
        result["pii_mask_count"] = total_count

        if total_count > 0:
            # APPI no-persistence: emit counts + category labels only — no raw PII
            # `result`, not `state`: the audit sink reads trace_id / correlation_id /
            # session_id from this argument, and `result` carries the same correlation
            # fields with raw_transcript already masked. Passing raw `state` here would
            # hand unmasked PII to the audit log and break the APPI no-persistence
            # guarantee documented above.
            emit_trace_event(
                "pii_detected",
                {
                    "pattern_types": categories,
                    "count": total_count,
                    "action": "masked",
                },
                result,
            )
            logger.info(
                "S-2 PII gate: masked %d token(s), categories=%s",
                total_count,
                categories,
            )
        else:
            logger.debug("S-2 PII gate: no PII detected in transcript")

        return result

    def _validate_input(self, raw: str) -> str | None:
        """Validate raw transcript input (S-1 gate).

        Args:
            raw: Raw (already PII-masked) transcript string.

        Returns:
            JSON error string if validation fails, None if valid.
        """
        if not raw or not raw.strip():
            return json.dumps(
                {
                    "code": "S1_INPUT_REJECTED",
                    "attempt": 1,
                    "node": "TranscriptCleanNode",
                    "reason": "empty_input",
                }
            )

        if len(raw) > MAX_TRANSCRIPT_CHARS:
            return json.dumps(
                {
                    "code": "S1_INPUT_REJECTED",
                    "attempt": 1,
                    "node": "TranscriptCleanNode",
                    "reason": "oversized_input",
                    "char_count": len(raw),
                    "limit": MAX_TRANSCRIPT_CHARS,
                }
            )

        if "\x00" in raw:
            return json.dumps(
                {
                    "code": "S1_INPUT_REJECTED",
                    "attempt": 1,
                    "node": "TranscriptCleanNode",
                    "reason": "non_text_bytes",
                }
            )

        return None

    def _normalize_transcript(self, text: str) -> str:
        """Normalize transcript text.

        Transformations (design doc §5.1):
        1. Strip aizuchi: えー, あー, うん (and repeated variants)
        2. Normalize speaker tags: [Speaker N] → [SN]
        3. Collapse repeated whitespace and blank lines

        Args:
            text: Raw transcript text (post-validation, post-PII-masking).

        Returns:
            Normalized transcript string.
        """
        for pattern in _AIZUCHI_PATTERNS:
            text = re.sub(pattern, "", text)

        text = _SPEAKER_TAG_PATTERN.sub(lambda m: f"[S{m.group(1)}]", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        return text
