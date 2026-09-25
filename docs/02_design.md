# Design Specification — CMN-C1-042
# Sales Call & Meeting Debrief Summary Agent

**Version:** 1.0 (draft)
**Stage:** ② Design
**Template ID:** `CMN-C1-042`
**Category:** Cat 1 — Generic, industry-agnostic
**L1 Base type:** `AgentBaseGraph` (L1-direct — ADR-002)
**L2 base agent:** None — L1-direct inheritance per 2026-05-18 policy (ADR-002)
**Author:** F-CuongTV19
**Last updated:** 2026-05-22

---

## §1 Template Overview

### §1.1 Purpose

CMN-C1-042 converts raw B2B sales meeting transcripts into three structured outputs
in a single LLM pass:

| Output key | Type | Description |
|---|---|---|
| `summary` | `str` | Structured meeting debrief (discussion points, objections, next actions, deal status) |
| `crm_fields` | `dict[str, Any]` | CRM-ready field values mapped to the injected CRM JSON Schema |
| `email_draft` | `str` | Follow-up email draft (subject + body, formality-controlled) |

All three outputs are produced by **one LLM call** — see §2.

### §1.2 Scope

**In scope:**
- Raw text transcript input (Zoom/Teams auto-transcription, copy-paste notes)
- Japanese business nuance classification before LLM call (`NuanceClassifyNode`)
- PII sanitization before LLM call (S-2 — APPI compliance)
- Single-pass structured extraction (summary + crm_fields + email_draft)
- CRM schema validation against injected JSON Schema (Salesforce / HubSpot / generic)
- Output formatting and S-3 content safety gate

**Out of scope:**
- Audio/video file ingestion (text only)
- Real-time streaming transcription
- Direct CRM API write (output is JSON payload for downstream integration)
- CRM OAuth / authentication

### §1.3 Output schema (LLM Pydantic model — NOT state)

```python
from pydantic import BaseModel
from typing import Any

class DebriefOutput(BaseModel):
    summary: str
    crm_fields: dict[str, Any]
    email_draft: str
```

`DebriefOutput` is used **only** inside `StructuredExtractNode` for LLM structured-output
parsing. It is instantiated, unpacked to primitives, then discarded — it is **never**
stored in `DebriefState`. See §4 for state safety rules.

### §1.4 ADR cross-references

| Decision | ADR | Summary |
|---|---|---|
| Single LLM call, one Pydantic schema | ADR-001 (issue #19) | Prevents dual-mode anti-pattern; 1/3 cost vs three-call design |
| `AgentBaseGraph` L1 direct, no L2 | ADR-002 (issue #19) | 2026-05-18 policy — no L2 base agents in `agent1000/agent-templates` |
| YAML-only CRM adapter | ADR-003 (issue #19) | Cat 1 zero-domain-assumption; citizen-developer configurable |

---

## §2 Single-Pass Contract

> ⚠️ **This is a hard architectural constraint. Do not add LLM calls to any other node.**

All three output keys (`summary`, `crm_fields`, `email_draft`) are produced by a
**single LLM invocation** inside `StructuredExtractNode` (main slot) using `DebriefOutput`
as the Pydantic structured-output schema.

**Why this matters:**

The moderator review (scaffold/#342, 2026-05-13) flagged multi-pass output generation
as a dual-mode anti-pattern violation — the same pattern that triggered CoE Critical
rejections on other templates. The approved v2.0 dev review (2026-05-20, score 29/30)
commits explicitly to single-pass. This decision is recorded in ADR-001 (issue #19).

**Consequences for implementers:**

- `TranscriptCleanNode`, `NuanceClassifyNode`, `CRMSchemaValidateNode`, `OutputFormatNode`,
  `SecurityGateNode` — zero LLM calls. All are deterministic transforms.
- `StructuredExtractNode` — exactly one LLM call. No retries within the node;
  retry logic is handled by the error handling wrapper (§9, issue #18).
- If a future change requires a second LLM call, a new ADR must be filed and the
  moderator + PM must approve before implementation.

**Single-pass prompt contract:**

The prompt instructs the LLM to populate all three output keys in one response.
For any CRM field not mentioned in the transcript, the LLM must output `null` —
never fabricate values (anti-hallucination instruction, mandatory in `prompts.yaml`).

---

## §3 L1 3-Slot Pipeline

### §3.1 Slot layout

```
[Input: raw_transcript (str) + config]
    │
    ▼
┌──────────────────────────────────────────────────┐
│  pre_process slot                                │
│  ① TranscriptCleanNode   — normalise input       │
│  ② NuanceClassifyNode    — JP nuance tagging     │
│  S-1: input validation (trust level gate)        │
│  S-2: PII detection + masking (APPI — mandatory) │
└───────────────────────┬──────────────────────────┘
                        │ annotated_transcript
                        ▼
┌──────────────────────────────────────────────────┐
│  main slot                                       │
│  ③ StructuredExtractNode                         │
│     ── SINGLE LLM CALL ──                        │
│     Pydantic: DebriefOutput                      │
│     {summary, crm_fields, email_draft}           │
│     methodology: BANT | MEDDIC (config-injected) │
│     anti-hallucination: null for absent fields   │
└───────────────────────┬──────────────────────────┘
                        │ raw extracted dict
                        ▼
┌──────────────────────────────────────────────────┐
│  post_process slot                               │
│  ④ CRMSchemaValidateNode — validate crm_fields   │
│     vs injected CRM JSON Schema                  │
│  ⑤ OutputFormatNode      — format final output   │
│  S-3: credential/injection scan + internal       │
│       data redaction (mandatory)                 │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
[Output: {summary: str, crm_fields: dict, email_draft: str}]
```

### §3.2 Node summary table

| # | Node | Slot | LLM? | Security gate |
|---|---|---|---|---|
| ① | `TranscriptCleanNode` | pre_process | No | S-1 (trust level), S-2 (PII mask) |
| ② | `NuanceClassifyNode` | pre_process | No | — |
| ③ | `StructuredExtractNode` | main | **Yes (1 call)** | — |
| ④ | `CRMSchemaValidateNode` | post_process | No | S-3 (schema conformance) |
| ⑤ | `OutputFormatNode` | post_process | No | S-3 (content safety + redaction) |

`SecurityGateNode` is not a separate pipeline node — S-1/S-2/S-3 gates are
implemented inline inside the slot sub-nodes (TranscriptClean for S-2, OutputFormat for S-3)
on the nodes that own each boundary (see §8).

### §3.3 Override method (new-gen)

All nodes — the three slot orchestrators **and** the five domain sub-nodes — inherit
`framework.nodes.function_node.FunctionNode` and override **`execute(self, state) -> dict`**
(per an internal implementation note, scaffold/#749). The backbone owns `__call__` (S-1 trust gate via
`required_trust_level`, timing, error capture, history) and the graph lifecycle
(`initialize → pre_process → main → post_process → finalize`); templates never implement
`_invoke_impl`, `__pre_invoke__`, or `_security_gate_*`. The slot orchestrators fold the
sub-nodes by calling their `execute()` directly (another template pattern).

```python
class TranscriptCleanNode(FunctionNode):  # type: ignore[misc]
    required_trust_level = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: dict) -> dict:
        # business logic; S-1 is enforced by the backbone on the slot orchestrator
        ...
```

Cross-reference: ADR-002 (issue #19), ADR-006 (S-1 on the node, enforced by BaseNode.__call__).


---

## §4 State Schema

### §4.1 State safety rules (mandatory — the framework contract, the security rules)

`DebriefState` is a **flat `TypedDict`**. The following are prohibited in state fields:

| Prohibited | Reason |
|---|---|
| `Pydantic BaseModel` instances | Breaks msgpack checkpoint serialization |
| `dataclass` instances | Same |
| `datetime` objects | Not msgpack-safe — use ISO 8601 `str` |
| `bytes` | Not msgpack-safe — use base64 `str` |
| JWT tokens, API keys, credentials | Stored plaintext in checkpoint DB |
| `InvocationContext` | Must travel via `config["configurable"]` only |

`DebriefOutput(BaseModel)` from §1.3 is used **only** inside `StructuredExtractNode`
for LLM output parsing. It is unpacked to primitives immediately after parsing and
discarded — it never touches state.

### §4.2 `DebriefState` definition

```python
# src/schemas/state.py
from __future__ import annotations
from typing import Any, TypedDict


class DebriefState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────────
    raw_transcript: str          # Raw meeting transcript (text only, max 50,000 chars)
    session_id: str              # Trace correlation ID (UUID, generated at entry)

    # ── Pre-process intermediates ─────────────────────────────────────────────
    cleaned_transcript: str      # Normalised transcript (aizuchi removed, speakers tagged)
    annotated_transcript: str    # cleaned_transcript + nuance annotations injected
    pii_masked: bool             # True if S-2 masked ≥1 PII token
    pii_mask_count: int          # Count of PII tokens masked (audit, S-4)

    # ── Main output (primitives only — DebriefOutput unpacked here) ───────────
    summary: str                 # Structured meeting summary
    crm_fields: str              # JSON str — CRM field values (msgpack-safe dict→str)
    email_draft: str             # Follow-up email draft (subject + body)

    # ── Post-process intermediates ────────────────────────────────────────────
    crm_validated: bool          # True if crm_fields passed CRM JSON Schema validation
    crm_validation_errors: str   # JSON str — list of validation error messages (empty="")
    output_json: str             # Final formatted output JSON str

    # ── Error handling ────────────────────────────────────────────────────────
    error: str                   # JSON str — {"code": str, "attempt": int} | ""
    retry_count: int             # LLM retry counter (reset per invocation)

    # ── Audit / trace (S-4) ──────────────────────────────────────────────────
    invocation_id: str           # UUID per invocation
    audit_input_hash: str        # SHA-256 of raw_transcript (no content in log)
    audit_output_hash: str       # SHA-256 of output_json (no content in log)
```

> **Note on `crm_fields: str`:** The CRM fields dict is stored as a JSON string
> (not `dict[str, Any]`) to satisfy msgpack checkpoint safety (the security rules,
> Rule 1.1). `CRMSchemaValidateNode` deserializes it, validates, then re-serializes.
> `OutputFormatNode` deserializes for final formatting.

### §4.3 `initial_state()` factory

All callers must construct state via `initial_state()` — never build the dict manually.

```python
def initial_state(raw_transcript: str, session_id: str) -> DebriefState:
    """
    Factory for a fully-initialized DebriefState.
    All fields explicitly set to safe empty defaults.
    """
    return DebriefState(
        raw_transcript=raw_transcript,
        session_id=session_id,
        cleaned_transcript="",
        annotated_transcript="",
        pii_masked=False,
        pii_mask_count=0,
        summary="",
        crm_fields="{}",
        email_draft="",
        crm_validated=False,
        crm_validation_errors="",
        output_json="",
        error="",
        retry_count=0,
        invocation_id="",        # populated by TranscriptCleanNode
        audit_input_hash="",     # populated by TranscriptCleanNode
        audit_output_hash="",    # populated by OutputFormatNode
    )
```

---

## §5 Node Specifications

All nodes (the three slot orchestrators and the five domain sub-nodes) inherit `FunctionNode` from
`framework.nodes.function_node`. Override method: `execute(self, state) -> dict` (see §3.3).

---

### §5.1 `TranscriptCleanNode`

**Slot:** `pre_process`
**File:** `src/nodes/transcript_clean.py`
**LLM:** No

**Purpose:** Normalize raw transcript; generate invocation ID and input hash;
enforce S-1 trust level gate; implement S-2 PII detection + masking (APPI).

**Input state fields:** `raw_transcript`, `session_id`

**Output state fields:** `cleaned_transcript`, `pii_masked`, `pii_mask_count`,
`invocation_id`, `audit_input_hash`

**S-1 gate (`required_trust_level`):** `TrustLevel.VERIFIED_EXTERNAL`
Transcripts contain customer data — anonymous callers must be rejected.

**S-2 gate (inline in TranscriptCleanNode.execute):** See §8.2 for PII categories and masking rules.
PII must be masked **before** `cleaned_transcript` is written — the LLM must never
receive unmasked PII.

**Processing steps:**
1. S-1 trust level enforced by the backbone `BaseNode.__call__` on the slot orchestrator (ADR-006)
2. S-2 PII detection + mask → write `pii_masked`, `pii_mask_count`
3. Validate input: reject if empty, if > 50,000 chars, or if non-text bytes detected
4. Generate `invocation_id` (UUID4)
5. Compute `audit_input_hash` (SHA-256 of raw_transcript, hex digest)
6. Normalize: strip aizuchi (「えー」「あー」「うん」), normalize speaker tags
   (e.g. `[Speaker 1]` → `[S1]`), collapse repeated whitespace
7. Write `cleaned_transcript`
8. S-4 trace events: `node_start`, `node_complete` (or `node_error`)

**Error policy:** On validation failure, set `state["error"]` = JSON error dict and
return — do not raise. Downstream nodes check `state["error"]` and short-circuit.

---

### §5.2 `NuanceClassifyNode`

**Slot:** `pre_process`
**File:** `src/nodes/nuance_classify.py`
**LLM:** No

**Purpose:** Tag Japanese polite-rejection and stalling phrases in the cleaned
transcript before the LLM call, so the LLM can correctly classify deal status
rather than misreading politeness as agreement.

**Input state fields:** `cleaned_transcript`
**Output state fields:** `annotated_transcript`

**Processing steps:**
1. Short-circuit if `state["error"]` is non-empty
2. Load nuance dictionary from `config/config.yaml` → `nuance.dictionary_path`
3. Scan `cleaned_transcript` for configured patterns; annotate inline:

| Pattern example | Annotation injected |
|---|---|
| 「前向きに検討します」 | `[NUANCE:DECLINE]` |
| 「持ち帰って検討します」 | `[NUANCE:DEFER]` |
| 「ちょっと難しいかもしれません」 | `[NUANCE:SOFT_DECLINE]` |
| 「ぜひ」「おっしゃる通り」 | `[NUANCE:POSITIVE]` |

4. Write `annotated_transcript` (= `cleaned_transcript` with inline annotations)
5. S-4 trace events

**Config key:** `nuance.dictionary_path` → path to YAML file with pattern → tag mapping.
Dictionary must be config-injectable — no hardcoded patterns in Python source.

---

### §5.3 `StructuredExtractNode`

**Slot:** `main`
**File:** `src/nodes/structured_extract.py`
**LLM:** **Yes — exactly one call**

**Purpose:** Single-pass LLM extraction producing all three output keys via
`DebriefOutput` Pydantic structured output schema.

**Input state fields:** `annotated_transcript`, `retry_count`
**Output state fields:** `summary`, `crm_fields` (JSON str), `email_draft`, `retry_count`

**Processing steps:**
1. Short-circuit if `state["error"]` is non-empty
2. Load prompt from `config/prompts.yaml` (EN) or `config/prompts_ja.yaml` (JP)
   based on `config.yaml` → `agent.output_language`
3. Select methodology block: `methodology.bant.*` or `methodology.meddic.*`
   based on `config.yaml` → `methodology`
4. Select email formality branch: `email.formal` / `email.semi_formal` / `email.casual`
   based on `config.yaml` → `email.formality_level`
5. Load CRM schema from `config.yaml` → `crm.schema_path` — inject field list into prompt
6. Call LLM with `DebriefOutput` as structured output schema (**one call**)
7. Unpack `DebriefOutput` to primitives:
   - `state["summary"]` = `output.summary` (str)
   - `state["crm_fields"]` = `json.dumps(output.crm_fields)` (str — msgpack safe)
   - `state["email_draft"]` = `output.email_draft` (str)
8. `DebriefOutput` instance discarded — never stored in state
9. S-4 trace events

**Error handling:** Delegated to wrapper (issue #18, §9). Node itself does not retry —
it sets `state["error"]` on LLM failure and returns. See §9.

---

### §5.4 `CRMSchemaValidateNode`

**Slot:** `post_process`
**File:** `src/nodes/crm_schema_validate.py`
**LLM:** No

**Purpose:** Validate `crm_fields` JSON against the injected CRM JSON Schema.
Implements S-3 custom extension for schema conformance (ADR-003).

**Input state fields:** `crm_fields` (JSON str)
**Output state fields:** `crm_validated`, `crm_validation_errors`, `crm_fields` (may update)

**Processing steps:**
1. Short-circuit if `state["error"]` is non-empty
2. Deserialize `crm_fields` JSON str → dict
3. Load CRM JSON Schema from `config.yaml` → `crm.schema_path`
4. Validate using `jsonschema.validate()` (or equivalent)
5. On validation errors: set invalid fields to `null` (graceful degradation —
   do NOT raise; field-level errors are non-fatal), record in `crm_validation_errors`
6. Re-serialize validated dict → `state["crm_fields"]` (JSON str)
7. Set `state["crm_validated"]` = True if zero errors, False otherwise
8. Emit S-4 trace event including `crm_validated` and error count

**Bundled CRM schemas** (see §7.1):
- `config/schemas/salesforce_opportunity.json`
- `config/schemas/hubspot_deal.json`

---

### §5.5 `OutputFormatNode`

**Slot:** `post_process`
**File:** `src/nodes/output_format.py`
**LLM:** No

**Purpose:** Format final output JSON; implement S-3 content safety and
internal data redaction gate.

**Input state fields:** `summary`, `crm_fields`, `email_draft`
**Output state fields:** `output_json`, `audit_output_hash`

**S-3 gate (inline in OutputFormatNode.execute):**
- Scan `summary` and `email_draft` for credential patterns (API keys, JWT tokens)
  → raise `SecurityViolationError` if found
- Redact internal project names and confidential references per
  `config.yaml` → `security.redact_patterns` (regex list)
- S-4 trace event if redaction applied

**Processing steps:**
1. Short-circuit if `state["error"]` is non-empty (still runs S-3 on partial output)
2. S-3 output gate
3. Assemble final output dict:
   ```python
   {
       "summary": state["summary"],
       "crm_fields": json.loads(state["crm_fields"]),
       "email_draft": state["email_draft"],
       "meta": {
           "crm_validated": state["crm_validated"],
           "pii_masked": state["pii_masked"],
           "invocation_id": state["invocation_id"],
       }
   }
   ```
4. Serialize to JSON str → `state["output_json"]`
5. Compute `audit_output_hash` (SHA-256 of `output_json`, hex digest)
6. S-4 trace events

---

## §6 Graph Composition

### §6.1 Framework contract

`Graph(AgentBaseGraph)` fills the three mandatory backbone slots (pre_process / main / post_process):
- `register_nodes()` sets `self._nodes` as a **side effect**, returns `None`
- `add_edges()` calls `self._sg.add_edge()` as **side effects**, returns `None`
- **No `execute()` override on the graph** — execution flows through
  `framework compile() → _compiled.invoke()` path

This is the correct pattern per FIND-ARCH-CMN094-01 and FIND-ARCH-CMN094-02
(both fixed in another template). another template's manual dispatch loop (`EDGE_ORDER` +
`execute()` override on the graph) is the older pattern — do not replicate it.

### §6.2 Graph skeleton

```python
# src/graph/graph.py
from __future__ import annotations
from typing import Any

from framework.graph.agent_base_graph import AgentBaseGraph
from src.nodes.pre_process_node import PreProcessNode
from src.nodes.main_node import MainNode
from src.nodes.post_process_node import PostProcessNode


class Graph(AgentBaseGraph):
    """New-gen Cat-1 composition — fills the three mandatory backbone slots with
    orchestrator FunctionNodes that fold the five domain sub-nodes (DI + execute()).
    The LLM client is injected here and forwarded to the main slot.

    Slot folds:
      pre_process  → PreProcessNode  (TranscriptClean S-2 + NuanceClassify)
      main         → MainNode        (StructuredExtract LLM + CRMSchemaValidate)
      post_process → PostProcessNode (OutputFormat — S-3 terminal gate; sole sink)

    Error propagation: sub-nodes self-guard on state["error"]; MainNode also
    short-circuits on a set error; PostProcessNode always runs (S-3 non-suppressible).
    The backbone owns compile()/invoke()/routing/initialize/finalize and the S-1
    trust gate (BaseNode.__call__, ADR-006). No src/agent.py.
    """

    def __init__(self, config: dict[str, Any] | None = None, llm_client: Any = None) -> None:
        self._llm_client = llm_client
        super().__init__(config)

    @property
    def name(self) -> str:
        return "cmn_c1_042"

    def register_nodes(self) -> None:
        super().register_nodes()  # backbone injects initialize / finalize
        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = MainNode(llm_client=self._llm_client)
        self._nodes["post_process"] = PostProcessNode()
```

### §6.3 Error propagation policy

Each node checks `state["error"]` at entry and short-circuits if non-empty,
**except `OutputFormatNode`** which always runs its S-3 gate to prevent unsafe
partial output from escaping the pipeline regardless of upstream error state.

```
TranscriptCleanNode  → sets state["error"] on S-1/S-2/validation failure
NuanceClassifyNode   → short-circuits if state["error"] set
StructuredExtractNode → short-circuits if state["error"] set;
                        sets state["error"] on LLM failure
CRMSchemaValidateNode → short-circuits if state["error"] set;
                        field-level schema errors are non-fatal (graceful degrade)
OutputFormatNode     → always runs S-3 gate; surfaces error in output_json
```


---

## §7 Configuration Surface

### §7.1 `config/config.yaml` — full schema

```yaml
# config/config.yaml — CMN-C1-042
# Runtime configuration for Sales Call & Meeting Debrief Summary Agent.
# Loaded once by src/graph/graph.py and the slot sub-nodes via yaml.safe_load().
# Per-request overrides are not supported — all settings are deployment-time.

# ── Agent behaviour ───────────────────────────────────────────────────────────
agent:
  output_language: "ja"           # "ja" | "en" — controls prompt selection
  max_transcript_chars: 50000     # S-1 hard ceiling (chars before any processing)

# ── Methodology ───────────────────────────────────────────────────────────────
# Selects which methodology block is injected into the extraction prompt.
methodology: "bant"               # "bant" | "meddic"

# ── CRM adapter ───────────────────────────────────────────────────────────────
# Path to CRM JSON Schema file. Schema is loaded at runtime and injected into
# the extraction prompt + used by CRMSchemaValidateNode.
# Bundled options:
#   config/schemas/salesforce_opportunity.json
#   config/schemas/hubspot_deal.json
# Citizen developers can supply their own schema file at this path.
crm:
  schema_path: "config/schemas/salesforce_opportunity.json"

# ── Email formality ───────────────────────────────────────────────────────────
email:
  formality_level: "formal"       # "formal" | "semi_formal" | "casual"
  # formal:     丁寧語 + 敬語 (enterprise B2B, first contact)
  # semi_formal: 丁寧語 only (warm leads, ongoing relationships)
  # casual:     平語 (intra-company, established clients)

# ── Japanese nuance classifier ────────────────────────────────────────────────
nuance:
  dictionary_path: "config/nuance_ja.yaml"
  # Keys in nuance_ja.yaml: pattern (regex str) → tag (DECLINE|DEFER|SOFT_DECLINE|POSITIVE)

# ── Security ──────────────────────────────────────────────────────────────────
security:
  s3_gate_enabled: true           # NON-SUPPRESSIBLE
  redact_patterns: []             # List of regex strings for internal data redaction
                                  # e.g. ["\\bProject[- ]?[A-Z]{2,}\\b"]

# ── LLM ──────────────────────────────────────────────────────────────────────
llm:
  model: "gpt-4o-mini"
  max_tokens: 2048
  temperature: 0.1                # Low temp: structured extraction, not creative
  timeout: 30                     # Per-request timeout passed to the LLM client
  retry:
    max_attempts: 3               # Max retry attempts (all failure modes — §9)
    backoff_factor: 1.0           # Seconds base for rate-limit exponential backoff
```

### §7.2 CRM schemas in `config/schemas/`

Two bundled JSON Schema files serve as reference implementations (ADR-003).
`CRMSchemaValidateNode` loads whichever path is set in `crm.schema_path`.

**`config/schemas/salesforce_opportunity.json`** — key fields:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "SalesforceOpportunity",
  "type": "object",
  "properties": {
    "Name":        { "type": ["string", "null"] },
    "StageName":   { "type": ["string", "null"],
                     "enum": ["Prospecting","Qualification","Needs Analysis",
                              "Value Proposition","Id. Decision Makers",
                              "Perception Analysis","Proposal/Price Quote",
                              "Negotiation/Review","Closed Won","Closed Lost", null] },
    "Amount":      { "type": ["number", "null"] },
    "NextStep":    { "type": ["string", "null"] },
    "Description": { "type": ["string", "null"] },
    "CloseDate":   { "type": ["string", "null"], "format": "date" }
  },
  "additionalProperties": false
}
```

**`config/schemas/hubspot_deal.json`** — key fields:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "HubSpotDeal",
  "type": "object",
  "properties": {
    "dealname":               { "type": ["string", "null"] },
    "dealstage":              { "type": ["string", "null"] },
    "amount":                 { "type": ["number", "null"] },
    "closedate":              { "type": ["string", "null"], "format": "date" },
    "notes_last_contacted":   { "type": ["string", "null"] },
    "hs_next_step":           { "type": ["string", "null"] }
  },
  "additionalProperties": false
}
```

### §7.3 Methodology YAML blocks in `config/prompts.yaml`

`StructuredExtractNode` selects a methodology block based on `config.yaml → methodology`.

```yaml
# config/prompts.yaml (EN) — methodology blocks

methodology:
  bant:
    fields: ["Budget", "Authority", "Need", "Timeline"]
    instruction: |
      Extract BANT qualification signals from the transcript.
      Map to crm_fields keys: budget, authority, need, timeline.
      If a signal was not discussed, output null for that key.
      Never fabricate values not present in the transcript.

  meddic:
    fields: ["Metrics", "Economic Buyer", "Decision Criteria",
             "Decision Process", "Identify Pain", "Champion"]
    instruction: |
      Extract MEDDIC qualification signals from the transcript.
      Map to crm_fields keys: metrics, economic_buyer, decision_criteria,
      decision_process, identify_pain, champion.
      If a signal was not discussed, output null for that key.
      Never fabricate values not present in the transcript.
```

### §7.4 Keigo branches in `config/prompts_ja.yaml`

`StructuredExtractNode` selects a formality branch based on
`config.yaml → email.formality_level`.

```yaml
# config/prompts_ja.yaml (JP) — email formality branches

email:
  formal: |
    以下のミーティングサマリーをもとに、丁寧語・敬語を使用したフォローアップメールを
    作成してください。初対面のエンタープライズ顧客向けの文体を使用してください。
    件名と本文を含めてください。
    サマリー: {summary}

  semi_formal: |
    以下のミーティングサマリーをもとに、丁寧語を使用したフォローアップメールを
    作成してください。関係が構築済みの顧客向けの文体を使用してください。
    件名と本文を含めてください。
    サマリー: {summary}

  casual: |
    以下のミーティングサマリーをもとに、平語を使用したフォローアップメールを
    作成してください。社内または親密な顧客向けの文体を使用してください。
    件名と本文を含めてください。
    サマリー: {summary}
```

---

## §8 Security Gates

### §8.1 S-1 — Authorization gate (`required_trust_level`, backbone-enforced)

**Owner:** Framework — `BaseNode.__call__` compares `caller_trust_level` against each node's `required_trust_level` before `execute()` runs (ADR-006). Templates declare the value; they do not implement the gate.
**Trust level set on:** `TranscriptCleanNode`

```python
class TranscriptCleanNode(FunctionNode):  # type: ignore[misc]
    required_trust_level = TrustLevel.VERIFIED_EXTERNAL
```

Sales transcripts contain customer PII and commercially sensitive deal information.
Anonymous callers (`TrustLevel.ANONYMOUS`) must be rejected before any processing.

**All three slot orchestrators** (`PreProcessNode`, `MainNode`, `PostProcessNode` — the nodes the
backbone registers and gates) declare `required_trust_level = TrustLevel.VERIFIED_EXTERNAL`; the
five domain sub-nodes folded inside them declare it too (defence-in-depth).

> S-1 is enforced by the backbone `BaseNode.__call__` upstream of `execute()` (ADR-006);
> templates only declare `required_trust_level`. The former scaffold/#907 ambiguity is
> resolved — see §3.3 and §10.

### §8.2 S-2 — PII detection gate (inline in TranscriptCleanNode.execute)

**Owner:** `TranscriptCleanNode`
**Mandatory** per `the security rules` — empty implementation
is a Critical security violation.

**PII categories to detect and mask** (source: `the security rules`, lines 88–94):

| Category | Examples | Mask token |
|---|---|---|
| Personal names | 田中太郎, John Smith | `[NAME]` |
| Addresses | 東京都渋谷区1-2-3, 123 Main St | `[ADDRESS]` |
| Phone numbers | 090-1234-5678, +81-3-1234-5678 | `[PHONE]` |
| Email addresses | taro@example.com | `[EMAIL]` |
| Government IDs | マイナンバー, Social Security numbers | `[GOV_ID]` |
| Financial identifiers | credit card numbers, bank account numbers | `[FINANCIAL]` |
| Medical record numbers | カルテ番号 | `[MEDICAL_ID]` |

**Masking rules:**
- Mask **before** writing `cleaned_transcript` — LLM must never receive unmasked PII
- Increment `state["pii_mask_count"]` per token masked
- Set `state["pii_masked"] = True` if ≥1 token masked
- Emit S-4 trace event: `"pii_detected"` with `{"pattern_types": [...], "count": N, "action": "masked"}`
- Raw PII must **not** appear in audit log, state snapshots, or S-4 trace payload
  (APPI no-persistence requirement)

**Implementation note:**
The S-2 PII gate is a functional (non-empty) implementation inside `TranscriptCleanNode.execute()` — never a pass-through.
In new-gen there is no separate `_security_gate_input()` method — S-2 lives inline in
`TranscriptCleanNode.execute()`. The equivalent violation would be a TranscriptCleanNode
that does not actually mask PII (a no-op pre-process). The implemented node masks PII and
records `pii_masked` / `pii_mask_count`, so the gate is functional and non-bypassable
(it is the first sub-node in the pre_process slot, which the backbone always runs first).

### §8.3 S-3 — Content safety filter (inline in OutputFormatNode.execute)

**Owner:** `OutputFormatNode` (primary), `CRMSchemaValidateNode` (schema conformance extension)
**Mandatory** per `the security rules` — empty implementation is a Critical security violation.

**Output checks:**

| Check | Action on detection |
|---|---|
| Credential patterns (API keys, JWT tokens, connection strings) | Raise `SecurityViolationError` — hard stop |
| Prompt injection markers (`[INST]`, `<\|system\|>`, etc.) | Raise `SecurityViolationError` |
| Internal project names / confidential references | Redact via `security.redact_patterns` regex list |
| Content safety (enterprise policy) | Sanitize + emit trace event |

`OutputFormatNode` runs S-3 **even when `state["error"]` is non-empty** —
no unsafe partial output escapes the pipeline (see §6.3).

### §8.4 S-4 — Structured audit trace (`_emit_trace_event`)

**Applies to:** ALL nodes — `TranscriptCleanNode`, `NuanceClassifyNode`,
`StructuredExtractNode`, `CRMSchemaValidateNode`, `OutputFormatNode`.
**Mandatory** per `the security rules` — silent failures are a security violation.

**Required trace event schema per node:**

```python
# On entry
self._emit_trace_event("node_start", {
    "node": self.__class__.__name__,
    "correlation_id": config["configurable"]["invocation_context"].correlation_id,
    "invocation_id": state.get("invocation_id", ""),
})

# On success
self._emit_trace_event("node_complete", {
    "node": self.__class__.__name__,
    "status": "success",
})

# On exception
self._emit_trace_event("node_error", {
    "node": self.__class__.__name__,
    "error": str(e),         # error type only — no PII, no transcript content
})
```

**Prohibited in trace payload:**
- Transcript content (raw or cleaned)
- PII tokens (names, phone, email, etc.)
- CRM field values
- Email draft content

**Permitted in trace payload:**
- `invocation_id`, `correlation_id`
- Node class name, status, error type string
- `audit_input_hash`, `audit_output_hash` (SHA-256 hex — content-free)
- `pii_mask_count`, `crm_validated`, `retry_count` (counts only)

**Audit retention:** Trace events are append-only. No trace event may be modified
or deleted after emission. Format is immutable per invocation.

### §8.5 S-5 — JWT detection at import time

Automatic — `BaseNode.__init_subclass__()` scans class definitions at import time.
No template implementation required. Do not include JWT patterns in class variables,
docstrings, or test fixtures (use `"mock-jwt-for-testing"` in tests).

---

## §9 Error Handling

### §9.1 Policy overview

`StructuredExtractNode` is the only LLM call in the pipeline. Three failure modes
each have distinct policies (see also issue #18):

| Failure mode | Detection | Policy |
|---|---|---|
| Malformed JSON / Pydantic schema validation failure | `ValidationError` or `json.JSONDecodeError` | Retry once with stricter prompt; on second failure → structured error state |
| Rate limit (HTTP 429) | `RateLimitError` or HTTP 429 | Exponential backoff + jitter, max `llm.retry_max` attempts; on exhaustion → structured error state |
| Timeout | `TimeoutError` raised by LLM client | Single retry with 50% transcript truncation; on second timeout → structured error state |

In all cases: **never raise an unhandled exception**. Always return a structured
error state. The caller receives a well-formed response with `state["error"]` set.

### §9.2 Error state format

```python
# state["error"] — JSON string stored in DebriefState
# Empty string ("") = no error

# On failure:
import json
state["error"] = json.dumps({
    "code": "LLM_SCHEMA_FAILURE",   # | "LLM_RATE_LIMIT" | "LLM_TIMEOUT"
    "attempt": 2,                    # Which attempt failed (1-indexed)
    "node": "StructuredExtractNode",
})
```

Error codes:

| Code | Trigger |
|---|---|
| `LLM_SCHEMA_FAILURE` | Pydantic `ValidationError` or `JSONDecodeError` on second attempt |
| `LLM_RATE_LIMIT` | HTTP 429 exhausted after `llm.retry_max` attempts |
| `LLM_TIMEOUT` | Timeout on second attempt (post-truncation retry) |
| `S1_INPUT_REJECTED` | S-1 validation failure (empty, oversized, non-text) |
| `S3_SECURITY_VIOLATION` | S-3 credential or injection pattern detected in output |

### §9.3 Retry logic (StructuredExtractNode wrapper)

```
Attempt 1
  ├─ Success → unpack DebriefOutput, continue pipeline
  └─ Failure
       ├─ ValidationError / JSONDecodeError
       │     → append stricter suffix to prompt
       │     → Attempt 2
       │          ├─ Success → continue
       │          └─ Failure → state["error"] = LLM_SCHEMA_FAILURE, return
       │
       ├─ RateLimitError (429)
       │     → backoff: sleep(base * 2^n + jitter), up to retry_max attempts
       │     → Exhausted → state["error"] = LLM_RATE_LIMIT, return
       │
       └─ TimeoutError
             → truncate annotated_transcript to 50% (tail-truncate)
             → Attempt 2
                  ├─ Success → continue
                  └─ Timeout again → state["error"] = LLM_TIMEOUT, return
```

`state["retry_count"]` is incremented on each retry attempt. S-4 trace event
emitted on each retry: `"llm_retry"` with `{"attempt": N, "reason": "..."}`.

### §9.4 Downstream node behaviour on error

Nodes downstream of `StructuredExtractNode` check `state["error"]` at entry
and short-circuit (return state unchanged) if non-empty — **except**
`OutputFormatNode`, which always runs its S-3 gate regardless (see §6.3).

`OutputFormatNode` surfaces the error in `output_json`:

```json
{
  "summary": "",
  "crm_fields": {},
  "email_draft": "",
  "error": {"code": "LLM_TIMEOUT", "attempt": 2, "node": "StructuredExtractNode"},
  "meta": {"invocation_id": "...", "crm_validated": false, "pii_masked": true}
}
```

---

## §10 Open Questions

### OQ-1 — BaseNode override contract (scaffold/#907)

**Filed:** 2026-05-22
**Status:** ✅ RESOLVED — new-gen contract confirmed (no longer awaiting CoE)

**RESOLVED (new-gen, sdk reference / agent-base-graph.md + node-contract.md):**

1. Node override method — **`execute(self, state) -> dict`** on `FunctionNode`
   (an internal implementation note, scaffold/#749). `_invoke_impl` is not used by templates;
   the backbone owns `__call__` and routing.

2. S-1 authorization — enforced by the backbone **`BaseNode.__call__`** before `execute()`,
   comparing `caller_trust_level` against each node's `required_trust_level` (ADR-006).
   Templates declare `required_trust_level`; they never call `__pre_invoke__` /
   `super().__pre_invoke__()`. The three mandatory slots (`pre_process`/`main`/`post_process`)
   are required by `compile()` (raises `MissingNodeError` if absent).

The earlier scaffold/#907 ambiguity no longer applies to this template — it is built to the
confirmed new-gen contract.
