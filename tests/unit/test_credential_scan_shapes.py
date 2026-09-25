"""The credential scan must catch the key formats providers issue TODAY.

Measured 2026-09-15: `sk-[A-Za-z0-9]{N,}` required N alnum characters immediately after
`sk-`, so any hyphen or underscore inside a key ended the match. Three of the four shapes
in use were missed -- including `sk-proj-...`, which is what OpenAI hands out by default.
A scan that misses the current default format is not a scan, and every test stayed green
because every fixture used the one old shape.

The patterns are read back out of the PRODUCTION source rather than restated here. A
copied constant cannot notice when the file it was copied from changes; this can.

🔴 The collector deliberately matches ANY literal containing `sk-`, not just the widened
shape. An earlier version matched only the widened shape, so a pattern reverting to the
old form simply vanished from the measurement and every assertion passed on the files that
had not reverted -- mutation-testing caught that on 3 of 20 repos. `_FILES` is asserted to
be fully covered for the same reason: a file going silent must fail, not disappear.
"""

import pathlib
import re

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Shapes providers issue. Labelled so the credential scan and a human reader both see
#: they are not live keys (the framework contract S-5).
_MUST_FLAG = {
    "classic": "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
    "project key (the current OpenAI default)": (
        "sk-proj-9dKq2LmXvR7TzYbN4pQwErTyUiOpAsDfGhJkLzXcVbNm1234"
    ),
    "service account": "sk-svcacct-abc123XYZ456def789GHI012jkl345MNO678pqr",
    "underscore in body": "sk-live_9dKq2LmXvR7TzYbN4pQwErTyUiOp",
}

#: A false positive costs a legitimate sender their answer, so this direction is tested too.
_MUST_NOT_FLAG = {
    "ordinary prose that contains the letters sk-": "please check the risk-assessment procedure",
    "a long hyphenated document name": "see risk-assessment-checklist-template-v2 for details",
    "a short real word": "the sk-lookup table is short",
    "an already-redacted value": "the log shows sk--------------------------- where the key was",
    "a masked value": "credential sk-____________________________ has been rotated",
}

_FILES = ['src/nodes/output_format.py']

#: Any string literal that mentions `sk-` AND carries regex machinery (a character class or
#: a quantifier). Matching the widened form only is what let three mutants live.
_LITERAL = re.compile(r"""["']([^"'\n]*sk-[^"'\n]*)["']""")


def _patterns() -> list[tuple[str, str]]:
    out = []
    for rel in _FILES:
        text = (_ROOT / rel).read_text(encoding="utf-8")
        for m in _LITERAL.finditer(text):
            rx = m.group(1)
            if "[" not in rx and "{" not in rx:
                continue                      # a plain string, not a pattern
            try:
                re.compile(rx)
            except re.error:
                continue                      # a fragment, not a whole pattern
            out.append((rel, rx))
    return out


def test_every_listed_file_still_contributes_a_pattern() -> None:
    """A file that goes silent must FAIL, not vanish from the measurement."""
    found = _patterns()
    assert found, f"no sk- pattern found in {_FILES} -- did the file move?"
    silent = sorted(set(_FILES) - {rel for rel, _ in found})
    assert not silent, f"these files no longer declare a credential pattern: {silent}"


@pytest.mark.parametrize("shape", sorted(_MUST_FLAG))
def test_each_real_key_shape_is_caught(shape: str) -> None:
    sample = _MUST_FLAG[shape]
    for rel, rx in _patterns():
        assert re.compile(rx).search(sample), f"{rel} misses the {shape} shape: {rx}"


@pytest.mark.parametrize("shape", sorted(_MUST_NOT_FLAG))
def test_no_ordinary_text_is_caught(shape: str) -> None:
    sample = _MUST_NOT_FLAG[shape]
    for rel, rx in _patterns():
        assert not re.compile(rx).search(sample), f"{rel} false-positives on {shape}: {rx}"
