# Test Specification — CMN-C1-042
# Sales Call & Meeting Debrief Summary Agent

**Version:** 1.0
**Stage:** ② Design → ③ Implementation
**Template ID:** `CMN-C1-042`
**Last updated:** 2026-05-25

---

## §1 Scope and Strategy

### §1.1 What is tested

This document covers all tests for `CMN-C1-042`. The test suite is organized
into three layers:

| Layer | Location | Purpose |
|---|---|---|
| **Unit** | `tests/unit/` | Each node and module in isolation — no real LLM calls |
| **Integration** | `tests/unit/test_agent.py` (E2E with mock LLM) | Full pipeline from `agent.run()` to `output_json` |
| **Proof-of-Boundary** | `tests/proof_of_boundary/` | Security gate contracts — S-1, S-2, S-3, S-4 |

### §1.2 What is NOT tested here

- Real LLM API calls (all LLM interactions use `MagicMock` in CI)
- CRM API write-back (out of scope per §1.2 of design doc)
- Audio/video ingestion (text only)

### §1.3 CI pass criteria

All of the following must be green before a PR can merge to `develop`:

1. `pytest tests/` — zero failures, zero errors
2. `python -c "from src.nodes.transcript_clean import TranscriptCleanNode"` — clean import
3. All `proof_of_boundary/` tests pass (no skips that are not explicitly version-gated)
4. No test may call a real external service (network access blocked in CI)

**Baseline:** 224 tests passing as of 2026-05-25 develop HEAD `be02937`.
After implementing `tests/proof_of_boundary/test_pb_cmn_c1_042.py` (issue #11):
expected total ≥ 232 tests.

---

## §2 Unit Test Cases

### §2.1 TranscriptCleanNode (`tests/unit/test_transcript_clean.py`)

| TC-ID | Test name | Assertion |
|---|---|---|
| TC-U-01 | `test_empty_input_rejected` | `state["error"]` set, code `S1_INPUT_REJECTED` |
| TC-U-02 | `test_oversized_input_rejected` | Input > 50,000 chars → `state["error"]` set |
| TC-U-03 | `test_exactly_max_chars_allowed` | Input = 50,000 chars → `state["error"]` == "" |
| TC-U-04 | `test_non_text_bytes_rejected` | Binary-mixed input → `state["error"]` set |
| TC-U-05 | `test_invocation_id_generated` | `state["invocation_id"]` is non-empty UUID string |
| TC-U-06 | `test_audit_hash_computed` | `state["audit_input_hash"]` is 64-char hex |
| TC-U-07 | `test_aizuchi_stripped` | 「えー」「あー」「うん」 absent from `cleaned_transcript` |
| TC-U-08 | `test_speaker_tag_normalised` | `[Speaker 1]` → `[S1]` in `cleaned_transcript` |
| TC-U-09 | `test_repeated_whitespace_collapsed` | Multiple spaces/blank lines collapsed |
| TC-U-10 | `test_short_circuit_on_error` | Pre-set `state["error"]` → node returns unchanged |
| TC-U-11 | `test_happy_path_all_fields_set` | All output fields populated on valid input |

**S-2 gate sub-cases** (`tests/unit/test_s2_pii_gate.py`):

| TC-ID | Class | Assertion |
|---|---|---|
| TC-U-S2-01 | `TestPhoneDetection` | JP mobile/landline/intl/bare-11digit → `[PHONE]` |
| TC-U-S2-02 | `TestEmailDetection` | RFC 5322 email → `[EMAIL]` |
| TC-U-S2-03 | `TestGovernmentIDDetection` | My Number (12-digit), SSN → `[GOV_ID]` |
| TC-U-S2-04 | `TestFinancialIDDetection` | CC 4×4, bare 16-digit, JP bank account → `[FINANCIAL]` |
| TC-U-S2-05 | `TestAddressDetection` | JP prefecture address, EN street address → `[ADDRESS]`; meeting rooms NOT masked |
| TC-U-S2-06 | `TestNameDetection` | JP 様/さん/氏 suffix → `[NAME]`; EN title-case pair → `[NAME]`; company bare name NOT masked |
| TC-U-S2-07 | `TestMedicalIDDetection` | カルテ番号, MR number → `[MEDICAL_ID]` |
| TC-U-S2-08 | `TestMultiplePIITypes` | All 7 categories in one transcript → all masked, count correct |
| TC-U-S2-09 | `TestS4TraceEvent` | `pii_detected` event emitted; payload has category labels only (no raw PII) |
| TC-U-S2-10 | `TestStateFields` | `pii_masked` / `pii_mask_count` set correctly; original state not mutated |

---

### §2.2 NuanceClassifyNode (`tests/unit/test_nuance_classify.py`)

| TC-ID | Test name | Assertion |
|---|---|---|
| TC-U-12 | `test_decline_pattern_annotated` | 「前向きに検討します」→ `[NUANCE:DECLINE]` injected |
| TC-U-13 | `test_defer_pattern_annotated` | 「持ち帰って検討します」→ `[NUANCE:DEFER]` injected |
| TC-U-14 | `test_soft_decline_pattern_annotated` | 「ちょっと難しいかもしれません」→ `[NUANCE:SOFT_DECLINE]` |
| TC-U-15 | `test_positive_pattern_annotated` | 「ぜひ」「おっしゃる通り」→ `[NUANCE:POSITIVE]` |
| TC-U-16 | `test_no_annotation_needed` | Transcript with no nuance patterns → unchanged |
| TC-U-17 | `test_empty_transcript` | Empty cleaned_transcript → annotated_transcript == "" |
| TC-U-18 | `test_missing_dictionary_file_graceful` | Bad path → no crash, transcript passed through |
| TC-U-19 | `test_multiple_patterns_all_annotated` | Two patterns in same transcript → both annotated |
| TC-U-20 | `test_short_circuit_on_error` | Pre-set error → node returns unchanged |

---

### §2.3 StructuredExtractNode (`tests/unit/test_structured_extract.py`)

| TC-ID | Class | Assertion |
|---|---|---|
| TC-U-21 | `TestStructuredExtractHappyPath` | BANT-complete transcript → all 3 outputs populated; LLM called exactly once |
| TC-U-22 | `TestMarkdownStripping` | JSON code fence in LLM response → stripped before parse |
| TC-U-23 | `TestErrorStates` | Invalid JSON → `LLM_SCHEMA_FAILURE`; upstream error → short-circuit |
| TC-U-24 | `TestMEDDICMethodology` | MEDDIC config → MEDDIC fields in prompt; BANT fields absent |
| TC-U-25 | `TestEmailFormality` | formal / semi_formal / casual blocks differ; injected into prompt |
| TC-U-26 | `TestHubSpotSchema` | HubSpot config → HubSpot field names in prompt; Salesforce absent |
| TC-U-27 | `TestBuildCrmFieldsBlock` | Salesforce/HubSpot/empty schema → correct field injection |
| TC-U-28 | `TestS4TraceEvents` | `node_start` / `node_complete` / `node_error` emitted; no PII in payload |
| TC-U-29 | `TestRetryWiring` | Bad JSON attempt 1 → retry → success; `node_retry` event emitted with correct fields; exhausted retries → error state |

---

### §2.4 CRMSchemaValidateNode (`tests/unit/test_crm_schema_validate.py`)

| TC-ID | Test name | Assertion |
|---|---|---|
| TC-U-30 | `test_valid_crm_fields_pass` | Valid Salesforce fields → `crm_validated == True`, no errors |
| TC-U-31 | `test_invalid_field_set_to_null` | Invalid field value → set to `null`, error recorded |
| TC-U-32 | `test_validation_errors_recorded` | `crm_validation_errors` list populated on failure |
| TC-U-33 | `test_additional_properties_removed` | Extra fields not in schema → removed |
| TC-U-34 | `test_missing_schema_file_sets_error` | Bad schema path → `state["error"]` set |
| TC-U-35 | `test_uses_bundled_salesforce_schema` | Default config → Salesforce schema loaded |
| TC-U-36 | `test_short_circuit_on_error` | Pre-set error → node returns unchanged |

---

### §2.5 OutputFormatNode (`tests/unit/test_output_format.py`)

| TC-ID | Test name | Assertion |
|---|---|---|
| TC-U-37 | `test_happy_path_output_json_set` | `output_json` is valid JSON with all 3 output keys |
| TC-U-38 | `test_meta_fields_included` | `meta.crm_validated`, `meta.pii_masked`, `meta.invocation_id` present |
| TC-U-39 | `test_audit_output_hash_set` | `audit_output_hash` is 64-char hex |
| TC-U-40 | `test_bearer_token_in_summary_raises` | Bearer token in summary → `SecurityViolationError` |
| TC-U-41 | `test_api_key_in_email_raises` | API key pattern in email → `SecurityViolationError` |
| TC-U-42 | `test_redact_patterns_applied` | `security.redact_patterns` config → patterns replaced in output |
| TC-U-43 | `test_s3_runs_even_when_error_set` | Error state → S-3 still runs; error surfaced in `output_json` |
| TC-U-44 | `test_crm_fields_deserialized_in_output` | `crm_fields` in output is dict (not JSON string) |

---

### §2.6 Graph, Agent, State (`tests/unit/test_graph.py`, `test_agent.py`, `test_state.py`)

| TC-ID | File | Assertion |
|---|---|---|
| TC-U-45 | `test_graph.py::TestSalesDebriefGraph` | 5 nodes registered; edge order correct; no `execute()` override on graph |
| TC-U-46 | `test_graph.py::TestBuildGraph` | `build_graph()` returns `AgentBaseGraph`; accepts `llm_client` and `config` |
| TC-U-47 | `test_agent.py::TestRun` | `agent.run()` returns dict with `summary`, `crm_fields`, `email_draft`; `session_id` generated |
| TC-U-48 | `test_agent.py::TestRun::test_run_error_state_fallback` | LLM failure → `run()` returns error dict (no raise) |
| TC-U-49 | `test_state.py::TestInitialState` | All 17 fields present; all primitives; `crm_fields` is valid JSON object string |

---

### §2.7 Retry module (`tests/unit/test_retry.py`)

| TC-ID | Class | Assertion |
|---|---|---|
| TC-U-50 | `TestRetrySchemaFailure` | Schema failure → retries once with strict prompt; second failure → `LLM_SCHEMA_FAILURE` error |
| TC-U-51 | `TestRetryRateLimit` | 429 → exponential backoff; exhausted → `LLM_RATE_LIMIT` error |
| TC-U-52 | `TestRetryTimeout` | Timeout → transcript truncated to 50%; second timeout → `LLM_TIMEOUT` error |
| TC-U-53 | `TestRetrySuccess` | Success first attempt → `retry_count` not incremented; no retry trace event |

---

### §2.8 S-4 Audit Trace (`tests/unit/test_audit_trace.py`)

| TC-ID | Class | Assertion |
|---|---|---|
| TC-U-54 | `TestGetCorrelationId` | `None` config → empty; `invocation_context` takes priority over `session_id` fallback |
| TC-U-55 | `TestTranscriptCleanNodeTrace` | `node_start` includes `correlation_id`; `node_complete` on success; `node_error` on validation failure |
| TC-U-56 | `TestNuanceClassifyNodeTrace` | `node_start` + `node_complete` emitted; `correlation_id` present |
| TC-U-57 | `TestCRMSchemaValidateNodeTrace` | `node_start` + `node_complete` emitted; `correlation_id` present |
| TC-U-58 | `TestOutputFormatNodeTrace` | `node_start` + `node_complete` emitted; `correlation_id` present |
| TC-U-59 | `TestStructuredExtractNodeTrace` | `node_start` includes `invocation_id` and `correlation_id` |

---

## §3 Integration Test Scenarios

Integration tests run the full pipeline via `SalesDebriefAgent.run()` with a mock
LLM client. No real API calls. Implemented in `tests/unit/test_agent.py`.

| TC-ID | Scenario | Fixture | Key assertions |
|---|---|---|---|
| TC-I-01 | Happy path — BANT complete | `BANT_COMPLETE_TRANSCRIPT` + `BANT_COMPLETE_LLM_RESPONSE` | `summary` non-empty; `crm_fields["Amount"] == 3_000_000`; `crm_validated == True`; `pii_masked == False`; no error |
| TC-I-02 | Error fallback — LLM failure | `BANT_COMPLETE_TRANSCRIPT` + mock raises `Exception` | `run()` returns dict; `error.code` present; no Python exception propagated |
| TC-I-03 | Security violation propagates | Mock LLM returns bearer token in summary | `run()` returns dict; `error` or exception surfaced cleanly |
| TC-I-04 | Session ID auto-generated | No `session_id` passed | `output["meta"]["invocation_id"]` is non-empty UUID string |
| TC-I-05 | Config override | `config_override={"methodology": "meddic"}` | Pipeline runs without error (no assertion on LLM content — mock) |

### §3.1 Missing integration scenarios (candidates for future expansion)

The following scenarios are not yet implemented in unit/integration tests and are
candidates for future issues:

| Gap | Notes |
|---|---|
| MEDDIC end-to-end with realistic fixture | `tests/fixtures/transcripts.py` only has BANT fixtures |
| JP keigo nuance → `[NUANCE:DECLINE]` surviving to LLM prompt | NuanceClassifyNode tested in isolation; no E2E fixture with keigo |
| HubSpot schema E2E | CRMSchemaValidateNode tested in isolation; no E2E agent run with HubSpot config |
| Retry E2E through agent | Retry tested at node level; `test_agent.py` only tests error fallback |

---

## §4 Proof-of-Boundary Test Cases

Proof-of-Boundary tests verify security gate contracts at the pipeline boundary.
They are deterministic (no real LLM calls) and mandatory for CI.

File: `tests/proof_of_boundary/test_pb_cmn_c1_042.py`

| PB-ID | Title | Scope | Method |
|---|---|---|---|
| PB-1 | S-1: empty input rejected | `TranscriptCleanNode.execute()` | Pass `raw_transcript=""` → assert `state["error"]` set, `error.code == "S1_INPUT_REJECTED"` |
| PB-2 | S-1: oversized input rejected | `TranscriptCleanNode.execute()` | Pass 50,001-char transcript → assert `state["error"]` set |
| PB-3 | S-2→S-3: PII absent from summary output | Full pipeline (mock LLM) | PII in transcript → assert PII tokens absent from `output["summary"]` |
| PB-4 | S-2→S-3: PII absent from email_draft output | Full pipeline (mock LLM) | PII in transcript → assert PII tokens absent from `output["email_draft"]` |
| PB-5 | Graceful degradation — LLM timeout | `SalesDebriefAgent.run()` | Mock LLM raises `TimeoutError` → assert `run()` returns dict (no raise); error code set |
| PB-6 | CRM schema: Salesforce fields in output | Full pipeline (mock LLM) | Salesforce config → `output["crm_fields"]` contains Salesforce-specific keys |
| PB-7 | CRM schema: HubSpot fields in output | Full pipeline (mock LLM) | HubSpot config → `output["crm_fields"]` contains HubSpot-specific keys |
| PB-8 | Invoke execution order | `BaseNode.__call__` → `execute()` sequence | ⚠️ Blocked — pending CoE clarification on `__pre_invoke__` dispatch (scaffold/#907, design doc §8.1). Add test once resolved: verify `__pre_invoke__` fires before `execute()`. See the framework contract PB-6 for the canonical definition of this boundary. |

> **Note — the framework contract PB-6 vs this spec PB-6**: the framework contract defines PB-6 as "Invoke execution order verification" (pre_invoke → gate_input → route → impl → gate_output → trace). This spec's PB-6 covers CRM schema (a different domain boundary). The the framework contract PB-6 boundary is tracked here as PB-8 pending scaffold/#907.

### §4.1 Existing PB tests

`tests/proof_of_boundary/test_pb_pii_gate.py` (`TestPBPIIGate`) — already implemented.
Verifies that PII is absent from `annotated_transcript` before it reaches `StructuredExtractNode`.

The new `test_pb_cmn_c1_042.py` extends coverage to the *output* side (S-3) and
to S-1 validation and graceful degradation.

---

## §5 Fixture Mapping

| Fixture | Location | Used by | Purpose |
|---|---|---|---|
| `BANT_COMPLETE_TRANSCRIPT` | `tests/fixtures/transcripts.py` | TC-U-21, TC-I-01 | Full BANT fields present; clean (no PII) |
| `BANT_COMPLETE_LLM_RESPONSE` | `tests/fixtures/transcripts.py` | TC-U-21, TC-I-01 | Mock LLM response for BANT-complete |
| `BANT_PARTIAL_TRANSCRIPT` | `tests/fixtures/transcripts.py` | TC-U-22 (null fields) | Partial BANT — tests null field handling |
| `BANT_PARTIAL_LLM_RESPONSE` | `tests/fixtures/transcripts.py` | TC-U-22 | Mock LLM with null fields |
| `MARKDOWN_WRAPPED_LLM_RESPONSE` | `tests/fixtures/transcripts.py` | TC-U-22 (markdown strip) | LLM response in code fence |
| `INVALID_JSON_LLM_RESPONSE` | `tests/fixtures/transcripts.py` | TC-U-23, TC-U-29, TC-U-50 | Triggers parse failure / retry |
| `MISSING_FIELD_LLM_RESPONSE` | `tests/fixtures/transcripts.py` | TC-U-23 | Triggers Pydantic ValidationError |
| `_PII_TRANSCRIPT` (inline) | `tests/proof_of_boundary/test_pb_pii_gate.py` | PB-PII-01..07 | PII across 7 categories |
| `_PII_TRANSCRIPT` (inline) | `tests/proof_of_boundary/test_pb_cmn_c1_042.py` | PB-3, PB-4 | PII survives to output check |

### §5.1 Fixture gaps (future work)

| Missing fixture | Needed for |
|---|---|
| `MEDDIC_COMPLETE_TRANSCRIPT` + LLM response | TC-I-05 (MEDDIC E2E), future integration tests |
| `KEIGO_DECLINE_TRANSCRIPT` | Keigo → `[NUANCE:DECLINE]` E2E path |
| `HUBSPOT_LLM_RESPONSE` | HubSpot E2E via `test_agent.py` |
| Oversized transcript (50,001 chars) | PB-2 (generated inline in PB test — no file needed) |

---

## §6 Coverage Targets

| Design doc section | Node / module | Target coverage | Status |
|---|---|---|---|
| §5.1 | `TranscriptCleanNode` | 100% of S-1 validation paths | ✅ TC-U-01..11 |
| §8.2 | S-2 PII gate (`_security_gate_input`) | All 7 PII categories + false-positive guards | ✅ TC-U-S2-01..10 |
| §5.2 | `NuanceClassifyNode` | All 4 annotation tags + no-match + error | ✅ TC-U-12..20 |
| §5.3 | `StructuredExtractNode` | Single-pass + BANT/MEDDIC + email formality + HubSpot + retry | ✅ TC-U-21..29 |
| §8.3 | S-3 gate (`_security_gate_output`) | Bearer token, API key, redact patterns, error-state passthrough | ✅ TC-U-40..43 |
| §5.4 | `CRMSchemaValidateNode` | Valid/invalid/extra/missing schema | ✅ TC-U-30..36 |
| §5.5 | `OutputFormatNode` | Output assembly, meta fields, hashing | ✅ TC-U-37..44 |
| §6 | `SalesDebriefGraph` | Node registration, edge order, no execute override | ✅ TC-U-45..46 |
| §9 | Retry policy | Schema/rate-limit/timeout × success/failure/exhausted | ✅ TC-U-50..53 |
| §8.4 | S-4 audit trace | All 5 nodes emit node_start + node_complete + node_error; no PII in payload | ✅ TC-U-54..59 |
| §4 | `DebriefState` | All 17 fields present; all primitives; immutable default | ✅ TC-U-49 |
| §1/§7 | `SalesDebriefAgent` | Config load, run(), error fallback, session_id | ✅ TC-U-47..48 |
| §8.1 | S-1 at pipeline boundary | Empty + oversized → rejected before LLM | ✅ PB-1, PB-2 |
| §8.2 | S-2 at pipeline boundary | PII absent from final output | ✅ PB-3, PB-4 |
| §9 | Graceful degradation | LLM timeout → no crash | ✅ PB-5 |
| §7.1 | CRM schema switching | Salesforce vs HubSpot field keys | ✅ PB-6, PB-7 |
| §8.1 | Invoke execution order (the framework contract PB-6) | pre_invoke → execute() sequence | ⚠️ PB-8 — blocked pending scaffold/#907 |
| §3 | MEDDIC E2E | Full pipeline with MEDDIC fixture | ⚠️ No E2E fixture yet |
| §5.2 | Keigo → LLM prompt E2E | `[NUANCE:DECLINE]` in annotated_transcript reaches LLM | ⚠️ No E2E fixture yet |

---

## §7 ADR Compliance Checks

These are not test cases but CI-visible invariants enforced by existing tests:

| ADR | Invariant | Enforced by |
|---|---|---|
| ADR-001 (single LLM call) | `llm_client.invoke.call_count == 1` on happy path | `TC-U-21::test_llm_invoked_exactly_once` |
| ADR-002 (L1 direct) | `SalesDebriefGraph` inherits `AgentBaseGraph`; no L2 import | `TC-U-45`, import check in CI |
| ADR-003 (YAML CRM adapter) | CRM schema loaded from config path; no hardcoded schema in Python | `TC-U-27`, `TC-U-35` |
| an internal implementation note | All nodes implement `execute()`, not `_invoke_impl()` | CI lint test `test_execute_method_signature()` (scaffold/#749) |
| State safety | All `DebriefState` fields are primitives | `TC-U-49::test_all_fields_are_primitive_types` |
| S-4 no-PII | No transcript content in any trace payload | `TC-U-S2-09::test_no_raw_pii_in_trace_payload` |

## Refused input — what the sender receives (shared contract, 2026-09-15)

Measured across the fleet with a real model: a message the framework's S-2 gate declined
came back as `status: error` carrying the generic line "No answer could be produced for
this request." `normalize_terminal_output()` raises on any status but SUCCESS, so the
runner discarded the whole envelope and the sender read **"agent failed"** — with nothing
to act on, and no reason to send anything different next time.

| Situation | What is returned | Why |
|---|---|---|
| S-2 declined the MESSAGE | `status: success`, `refusal_kind: "input"`, a sentence naming what to change, plus the trailer | The sender is legitimate and holds something they can fix; they only learn that if the reply reaches them |
| The agent has its own refusal wording | That wording, not the shared sentence | "The shipment could not be classified" says which step stopped; the generic line does not |
| S-1 denied the CALLER | `status: error`, `refusal_kind: "trust"`, the refusal and nothing else | A caller not permitted to invoke the agent must not be told what it is for |
| S-3 blocked the agent's OWN output | unchanged — `status: error` | The agent produced something its output gate would not pass. The sender can do nothing with that, and must not be invited to retry |
| The agent genuinely broke | unchanged — `status: error` | The one signal that says this is an operations problem |

Nothing downstream reads `status` to detect a refusal any more: the envelope names the
refusal in `refusal_kind`. A contract that could only be read by the symptom it was fixing
was not a contract.

Enforced by `tests/unit/test_disclaimer_always_present.py` —
`test_a_refused_MESSAGE_is_delivered_and_says_what_to_change`,
`test_a_REAL_failure_is_still_an_error` (its control), and
`test_the_gate_token_is_matched_as_a_whole_token`.

## Credential scan — key shapes (added 2026-09-15)

| Input | Expected | Why |
|---|---|---|
| `sk-` + 20 alnum (classic) | flagged | the only shape the suite used to exercise |
| `sk-proj-...` | flagged | **OpenAI's current default**. Missed before this round: the pattern required N alnum characters straight after `sk-`, so the hyphen ended the match |
| `sk-svcacct-...` | flagged | service-account keys, same mechanism |
| `sk-live_...` | flagged | underscore in the body, same mechanism |
| `risk-assessment-checklist-template-v2` | NOT flagged | contains the literal `sk-`; a false positive costs a legitimate sender their answer |
| `sk-` followed by dashes/underscores only | NOT flagged | an already-redacted value — the sender did the right thing |

Enforced by `tests/unit/test_credential_scan_shapes.py`, which reads the patterns back out
of the production source instead of restating them, and fails if a listed file stops
declaring one. Counting patterns is not measuring: the previous set had four patterns and
missed three of the four shapes in use.
