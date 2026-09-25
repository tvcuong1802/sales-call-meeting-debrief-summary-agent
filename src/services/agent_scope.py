"""What THIS agent's output must not be used for, and how it reads a request.

Kept in its own module so the wording lives beside the agent it describes, while the
MECHANISM stays byte-identical fleet-wide in ``disclaimer.py`` and ``input_intake.py``.
A generic "AI-generated draft" is true of every template here and tells a reader nothing
they can act on; this names the decisions the output must not stand in for.
"""

from __future__ import annotations

SCOPE_EN = (
    "Reference only: a debrief drafted from the transcript supplied. It is not a record of "
    "what was agreed, not a commitment on the customer's behalf, and the CRM values and "
    "follow-up email are drafts. Confirm against your own notes before sending anything or "
    "updating a record."
)

SCOPE_JA = (
    "参考情報です。提供された議事録から作成した要約案であり、合意事項の記録でも、お客様に代わ"
    "る確約でもありません。CRM 項目およびフォローアップメールは下書きです。送信や記録の更新の"
    "前に、ご自身のメモと照合してください。"
)


# How this agent's answers are read for language, on top of the shared defaults in
# disclaimer.DEFAULT_LANGUAGE_POLICY. Empty means the defaults were measured to hold --
# see deploy/disclaimer_cases.json for the cases they were measured on.
LANGUAGE_POLICY: dict[str, object] = {
    # CRM field names are the customer's schema keys -- Salesforce calls them
    # StageName, NextStep, CloseDate -- and they stay in Latin inside a Japanese
    # debrief the way METI and HITL do elsewhere. Without this the field list alone
    # crossed the bilingual threshold and a fully Japanese debrief asked for an
    # English notice too.
    "identifiers": (
        "Name",
        "StageName",
        "Amount",
        "NextStep",
        "Description",
        "CloseDate",
        "Acme Corp",
    ),
}


# What the intake call reads out of a free-form request. No `fields` are declared: the
# transcript is parsed by structured_extract with a prompt written for this domain, and a
# second extractor would be a second source of truth for the same values. What the call
# adds is the language of the debrief -- a request typed in romanised Japanese is entirely
# Latin, and reading the characters gets that reader wrong.
INTAKE_POLICY: dict[str, object] = {
    "languages": ("en", "ja"),
    "default_language": "en",
    "fields": {},
    "capabilities": (
        "Summarise a sales meeting transcript into discussion points, objections and next actions",
        "Map what was said onto the CRM fields your schema defines",
        "Draft a follow-up email from the meeting",
    ),
    "examples": (
        {
            "message": "kono uchiawase no giji roku wo youyaku shite kudasai",
            "expect": {"language": "ja", "fields": {}, "fits": "yes", "suggestion": None},
        },
        {
            "message": "この商談の議事録を要約して、CRM 項目とフォローアップメールも作ってください。",
            "expect": {"language": "ja", "fields": {}, "fits": "yes", "suggestion": None},
        },
        {
            "message": "Here is the transcript from today's call with Acme -- write me the debrief.",
            "expect": {"language": "en", "fields": {}, "fits": "yes", "suggestion": None},
        },
        {
            # Adjacent, not ours: about the meeting, but asking the agent to act on it.
            "message": "Send the follow-up email to the customer and update the CRM record.",
            "expect": {"language": "en", "fields": {}, "fits": "no", "suggestion": 3},
        },
    ),
}
