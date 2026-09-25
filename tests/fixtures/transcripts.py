"""
Golden dataset fixtures for StructuredExtractNode tests — CMN-C1-042.

Realistic B2B sales meeting transcript samples with expected extraction outputs.
Used by test_structured_extract.py for mock LLM response verification.
"""

from __future__ import annotations

import json

# ── Fixture 1: BANT-complete transcript (all fields present) ─────────────────

BANT_COMPLETE_TRANSCRIPT = """\
[S1] 本日はお時間いただきありがとうございます。弊社のCRMソリューションについてご説明させてください。
[S2] こちらこそよろしくお願いします。予算については上半期で300万円を確保しています。
[S1] ありがとうございます。導入のご決定はどなたが？
[S2] 私と情報システム部長の二名で最終判断します。6月末までに導入を完了したい考えです。
[S1] 承知しました。現状の課題をお聞かせください。
[S2] 営業日報の集計に毎週5時間かかっています。これを自動化したいです。
[S1] それであれば弊社のレポート自動化機能が最適です。来週デモをご用意できます。
[S2] ぜひお願いします。"""

BANT_COMPLETE_EXPECTED_CRM = {
    "Name": None,
    "StageName": "Needs Analysis",
    "Amount": 3000000,
    "NextStep": "来週デモを実施",
    "Description": "CRMソリューション導入検討。予算300万円、6月末導入希望。営業日報集計の自動化が主な課題。",
    "CloseDate": "2026-06-30",
}

BANT_COMPLETE_LLM_RESPONSE = json.dumps(
    {
        "summary": "予算300万円（上半期確保済み）、意思決定者は担当者＋情シス部長の2名。6月末導入目標。現課題：営業日報集計に週5時間。ネクストステップ：来週デモ実施。",
        "crm_fields": BANT_COMPLETE_EXPECTED_CRM,
        "email_draft": "Subject: 先日のご面談御礼とデモのご案内\n\n○○様\n\n本日はお時間をいただきありがとうございました。\n来週のデモについてご連絡いたします。\n\n何卒よろしくお願いいたします。",
    }
)

# ── Fixture 2: Partial transcript (many fields absent — null test) ────────────

BANT_PARTIAL_TRANSCRIPT = """\
[S1] 御社の課題についてもう少し詳しく教えていただけますか。
[S2] 在庫管理の効率化を検討しています。
[S1] 具体的にどのような問題がありますか。
[S2] 手作業が多く、ミスが発生しています。予算や時期はまだ決まっていません。"""

BANT_PARTIAL_LLM_RESPONSE = json.dumps(
    {
        "summary": "在庫管理効率化の初期相談。手作業によるミス発生が課題。予算・導入時期・意思決定者は未確定。",
        "crm_fields": {
            "Name": None,
            "StageName": "Prospecting",
            "Amount": None,
            "NextStep": None,
            "Description": "在庫管理効率化の検討。手作業ミスが課題。予算・時期未定。",
            "CloseDate": None,
        },
        "email_draft": "Subject: 先日のご相談御礼\n\n○○様\n\n本日はお時間をいただきありがとうございました。\nご検討のほどよろしくお願いいたします。",
    }
)

# ── Fixture 3: Markdown-wrapped LLM response (code fence stripping test) ─────

MARKDOWN_WRAPPED_LLM_RESPONSE = (
    "```json\n"
    + json.dumps(
        {
            "summary": "テスト用サマリー",
            "crm_fields": {
                "Name": "テスト案件",
                "StageName": "Qualification",
                "Amount": None,
                "NextStep": None,
                "Description": None,
                "CloseDate": None,
            },
            "email_draft": "Subject: テスト\n\nテスト本文",
        }
    )
    + "\n```"
)

# ── Fixture 4: Invalid JSON response (parse failure test) ─────────────────────

INVALID_JSON_LLM_RESPONSE = "I'm sorry, I cannot extract structured data from this transcript."

# ── Fixture 5: Missing field response (Pydantic validation failure test) ──────

MISSING_FIELD_LLM_RESPONSE = json.dumps(
    {
        "summary": "テスト",
        # crm_fields missing — Pydantic ValidationError expected
        "email_draft": "Subject: テスト\n\n本文",
    }
)
