"""What the reader receives: sections, in their language, and a word when nothing was read.

`result` used to be the raw JSON blob. Marketplace shows it verbatim in a chat surface,
so three different things -- a debrief, CRM values and an email draft -- arrived as one
machine string. This is the renderer that replaced it, and it had no test at all.
"""

from src.nodes.output_format import _CHROME, _as_markdown

_FULL = {
    "summary": "Acme confirmed budget is approved.",
    "crm_fields": {"Name": "Acme Corp", "StageName": "Qualification"},
    "email_draft": "Subject: Migration plan\n\nThank you for your time.",
    "meta": {"crm_validation_errors": []},
}


# --- the three parts are told apart --------------------------------------------------
def test_each_part_gets_its_own_section():
    out = _as_markdown(_FULL, "en")
    assert "## Debrief" in out and "## CRM fields" in out and "## Follow-up email draft" in out
    assert out.index("## Debrief") < out.index("## CRM fields") < out.index("## Follow-up email draft")


def test_crm_values_are_listed_not_dumped():
    out = _as_markdown(_FULL, "en")
    assert "- **Name**: Acme Corp" in out and "- **StageName**: Qualification" in out
    assert "{'Name'" not in out and '{"Name"' not in out


def test_validation_errors_are_shown_because_they_change_what_a_reader_may_paste():
    # They were recorded in meta and never printed.
    report = {**_FULL, "meta": {"crm_validation_errors": ["Amount is not a number"]}}
    out = _as_markdown(report, "en")
    assert "## Needs attention" in out and "- Amount is not a number" in out


def test_no_attention_section_when_there_is_nothing_to_attend_to():
    assert "## Needs attention" not in _as_markdown(_FULL, "en")


# --- language ------------------------------------------------------------------------
def test_a_japanese_debrief_carries_japanese_headings():
    out = _as_markdown(_FULL, "ja")
    assert "## 商談サマリ" in out and "## CRM 項目" in out and "## フォローアップメール案" in out
    assert "## Debrief" not in out and "## Follow-up email draft" not in out


def test_an_unknown_language_reads_as_english_rather_than_half_a_page():
    for language in ("fr", "mixed", None):
        out = _as_markdown(_FULL, language)
        assert "## Debrief" in out and "## 商談サマリ" not in out


def test_the_two_label_sets_declare_the_same_keys():
    # A key in one language and not the other raises KeyError at render time.
    assert set(_CHROME["en"]) == set(_CHROME["ja"])


# --- nothing was read -----------------------------------------------------------------
def test_an_empty_result_says_no_debrief_was_produced_not_four_empty_sections():
    """Four empty sections read as findings ABOUT the transcript.

    "No CRM fields could be filled from this transcript" is a statement about the data;
    "the model never ran" is a statement about the agent, and a reader has no way to tell
    those apart. That is the expensive shape: not a crash, a confident wrong answer.
    """
    empty = {"summary": "", "crm_fields": {}, "email_draft": "", "meta": {}}
    out = _as_markdown(empty, "en")
    assert "could not be generated" in out
    assert "## CRM fields" not in out


def test_the_no_debrief_message_is_bilingual_when_no_language_was_determined():
    empty = {"summary": "", "crm_fields": {}, "email_draft": "", "meta": {}}
    out = _as_markdown(empty, "")
    assert "could not be generated" in out and "要約を作成できませんでした" in out


def test_the_no_debrief_message_follows_a_known_language():
    empty = {"summary": "", "crm_fields": {}, "email_draft": "", "meta": {}}
    out = _as_markdown(empty, "ja")
    assert "要約を作成できませんでした" in out and "could not be generated" not in out


# --- a missing email is stated, not silently omitted -----------------------------------
def test_a_missing_email_draft_says_so():
    report = {**_FULL, "email_draft": ""}
    out = _as_markdown(report, "en")
    assert "## Follow-up email draft" in out and "No follow-up email was drafted" in out
