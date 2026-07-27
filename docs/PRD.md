# ClaimCourt Product Requirements Document

## 1. Product Decision

**Product:** ClaimCourt: A Private Local AI Evidence Court

**Competition:** AMD AI DevMaster Hackathon 2026, Track 2: Development & Local Deployment of Private AI Agents

**North-star outcome:** Win Track 2 by making the strongest possible case that private AI agents can be useful, trustworthy, performant, and locally deployable on AMD Radeon GPU + ROCm.

**One-sentence pitch:** ClaimCourt turns private documents into cited verdicts, contradiction timelines, and approval-gated decision briefs using a local AMD GPU model.

**Ten-second demo test:** A viewer should immediately understand: "Instead of asking a black-box chatbot what a contract says, I can put my private evidence on trial and inspect the reasoning."

## 2. Problem

Disputes over commitments, approvals, deadlines, and obligations rarely live in one document. A customer claims a sales promise, procurement points to the contract, and engineering cites conditional meeting notes. Conventional document chat often returns a fluent answer without distinguishing binding evidence, marketing, draft language, exceptions, or missing records.

The target user needs a private, auditable factual brief before acting. They do not need an autonomous legal agent or an overconfident answer.

## 3. Target User And Job

### Primary user

An operations, procurement, customer-success, legal-operations, or compliance lead facing a document-backed dispute.

### Primary job

"Before I negotiate, escalate, or approve a response, show me exactly what our private record supports, what contradicts it, and what evidence is missing."

### Canonical demo question

`Did the vendor contractually commit to 99.9% uptime?`

### Correct outcome

`Insufficient contractual commitment.` Sales language is non-binding marketing; meeting notes state a conditional future target; the agreement lacks a signed SLA commitment. Recommended action: request a signed SLA addendum.

## 4. Product Principles

1. **Evidence before inference.** The product must surface evidence before it asks a model to draw conclusions.
2. **Citations are a hard boundary.** A role may cite only IDs from the current retrieved packet. Unknown citations are removed and recorded as a model-quality failure.
3. **Insufficient evidence is a successful answer.** The system must not invent certainty.
4. **One model, controlled roles.** Three fixed serial roles use the same local instruction model. This is auditable orchestration, not performative autonomy.
5. **Privacy is a user-visible behavior.** The app states where data stays and requires explicit approval before a report file is written.
6. **AMD performance is observable.** The product shows model/runtime configuration and latency metrics rather than asserting GPU acceleration in prose.
7. **Failure is explicit.** Missing model endpoint, model download failure, parser failure, no evidence, and missing approval produce clear states and actions.

## 5. Scope And Non-Goals

### Required for the final hackathon build

| Capability | Requirement | Competition coverage |
| --- | --- | --- |
| Local evidence intake | PDF, DOCX, TXT, Markdown, EML; parse failures are visible | Local RAG |
| Local retrieval | Chunk, score, retrieve, and preserve stable citation IDs | Local RAG + tool use |
| Controlled roles | Prosecution, defense, judge; strict JSON schema | Multi-step planning |
| Verdict surface | Verdict, confidence, reasoning, citations, contradictions, timeline, missing evidence, next action | Functional completeness |
| Case memory | Workspace evidence index persists locally with source hashes and changed-file version events | Local memory |
| Approval gate | No local brief is written until the user approves | Permission + privacy |
| Local inference | Qwen3-class open model served by vLLM on Radeon/ROCm | AMD/ROCm |
| Performance proof | Model, dtype/quantization, first token, total latency, tokens/s | AMD/ROCm |
| Stable demo | Synthetic SLA corpus, expected verdict, reproducible script | Demo reliability |

### Explicit non-goals

- Legal advice or legally binding verdicts.
- Background autonomous actions, document sharing, or external email sending.
- Public hosted knowledge base behavior.
- Fine-tuning, model training, or real multi-model collaboration.
- Any closed remote LLM API as a core function.

## 6. Final Product Shape

The final submission opens directly into the **Case Room**, not a marketing landing page.

### A. Case Room: first viewport

- Header: ClaimCourt identity, `Private workspace` status, model/runtime state, and Radeon/ROCm indicator.
- Left intake rail: evidence sources, parse status, source count, and case controls.
- Main center: disputed claim editor and one decisive `Open Evidence Court` command.
- Right rail: privacy boundary, case memory state, and permission state.

### B. Evidence Court record

- A fixed-height workflow strip shows `Retrieve -> Prosecution -> Defense -> Judge` with status and elapsed time.
- Verdict banner uses three explicit states: Supported, Contradicted, Insufficient Evidence.
- Evidence cards show citation ID, source, exact excerpt, relevance, source page/date where available, and a copy citation control.
- Prosecution and defense are paired for comparison, not hidden behind chat logs.
- Timeline visualizes dated assertions and flags contradictory/conditional events.
- Missing evidence is prominent because it changes the human next action.

### C. Decision brief approval

- The user sees a preview and must select `Approve local export` before a Markdown/PDF file is written.
- The exported report contains the complete evidence list, contradictions, missing evidence, and the exact verdict.
- If the user declines, no file is created.

### D. AMD performance drawer

- GPU model/architecture, ROCm/HIP version, vLLM version, model name, dtype/quantization, max context, VRAM utilization.
- First-token latency, end-to-end latency, completion tokens, tokens/s, and optional warmed vs cold request comparison.
- Any unavailable measurement is labeled unavailable; no fabricated performance number.

## 7. Functional Requirements

### FR-0: Semantic memory finder

- The user may describe an artifact or fact imprecisely; exact filenames are not required.
- A local intent planner emits the inferred intent, artifact types, topics, entities, time hints, expanded concepts, and search scope before retrieval.
- File discovery combines semantic text relevance with file-name/type and locator signals; every result displays human-readable match reasons.
- Similar copies and version-like files are grouped into a file family while distinct evidence excerpts remain visible.
- The UI must show confidence and allow the user to inspect the search plan instead of presenting an opaque nearest-neighbor result.
- If a local Ollama or vLLM embedding endpoint is configured, embedding similarity is blended with lexical retrieval; endpoint failure must fall back to deterministic local retrieval.
- Query plans and result identities are persisted locally so a case owner can inspect or replay prior searches without re-uploading evidence.

### FR-1: Evidence intake

- User can load the stable demo corpus in one command.
- User can upload supported private file types.
- System emits a per-file success/error state and never silently omits an unreadable file.
- System assigns source IDs and citation IDs that remain stable within the case.
- System persists a local source manifest and records a `changed` version event when a previously indexed path has a different SHA-256.

### FR-2: Retrieval tool

- System converts a claim into a local search query.
- System retrieves a configurable top-k packet.
- System renders all packet evidence before rendering the final verdict.
- Search never calls a remote retrieval or embedding API.

### FR-3: Controlled court workflow

- Prosecution identifies the strongest support and returns `role`, `position`, `citations`, and observations.
- Defense identifies exceptions, non-binding wording, missing conditions, and contradictions.
- Judge receives only the retrieved packet and the two structured arguments.
- Judge returns JSON with: `claim`, `verdict`, `confidence`, `reasoning`, `evidence_citations`, `contradictions`, `timeline_events`, `missing_evidence`, and `recommended_next_action`.
- System validates JSON and strips citations/timeline source IDs outside the allowed packet.

### FR-4: Verdict semantics

- `supported`: retrieved evidence directly supports the stated claim.
- `contradicted`: retrieved evidence directly refutes the stated claim.
- `insufficient_evidence`: record lacks enough binding/direct evidence to decide safely.
- Confidence expresses evidence sufficiency, not legal certainty.

### FR-5: Privacy and permissions

- Display a visible local-processing statement before evidence intake.
- Never send source contents to a closed model provider.
- Local brief writing requires an explicit unchecked-by-default approval.
- Public network exposure of the UI must be labeled unsafe for real private evidence.

### FR-6: Resilience

- When vLLM is unavailable, show `Local model unavailable` with endpoint diagnostics and retry action.
- Deterministic fallback may be enabled only for the synthetic demo and must be visibly labeled `demo fallback`, never `local vLLM`.
- When model artifact download is blocked, report the actual network/model state and offer offline artifact upload instructions.
- The product must continue to display already retrieved evidence and allow no-model inspection.

### FR-7: Export

- Approval gate writes a local Markdown brief.
- The brief includes generation time, verdict, citations, quoted excerpts, contradictions, missing evidence, and next action.
- PDF export is a stretch goal only after Markdown export is reliable.

## 8. Technical Requirements

### Local model path

- Primary model: Qwen3-8B or a compatible Qwen3-class instruction model.
- Server: local vLLM OpenAI-compatible endpoint at `http://127.0.0.1:8000/v1`.
- Target Radeon Cloud image: `ROCm vLLM-dev (Navi)`.
- Dtype: BF16 on Radeon Pro W7900-class VRAM; supported AWQ/GPTQ only if needed and measured.
- One model instance serves all three roles serially to keep VRAM use and behavior predictable.
- Semantic retrieval may use a separate local BGE-small embedding runner on port 8001; the judge model and embedding model have independent endpoints and telemetry.

### Deployment requirement

- Verify `rocminfo`, PyTorch ROCm availability, HIP version, and vLLM version before model work.
- Preserve artifacts outside ephemeral cloud runtime before restarting services.
- Prefer an SSH local port forward for private UI access when provider web routing is unreliable.

### Reliability requirement

- Demo runs must have a smoke test for parser, retrieval, verdict schema, citation validation, vLLM endpoint health, and UI health.
- A model download must be pre-staged from a trusted artifact source before recording the video. Do not rely on an unverified cloud egress path.

## 9. User Stories And Acceptance Criteria

### Story 1: Evaluate an SLA claim

As a procurement lead, I load five private documents and ask whether the vendor committed to 99.9% uptime.

**Acceptance criteria**
- Result is `Insufficient Evidence` for the supplied corpus.
- At least one cited agreement excerpt says a separate signed addendum is required.
- Sales language and the conditional meeting target appear as distinct evidence, not equivalent commitments.
- Next action requests a signed SLA addendum.

### Story 2: Challenge the answer

As a skeptical reviewer, I inspect every material citation and see where prosecution and defense disagree.

**Acceptance criteria**
- Every displayed citation resolves to an excerpt in the retrieved packet.
- No model-created source ID can appear in the report.
- Timeline includes dated meeting/customer-request events when dates exist.

### Story 3: Export only after consent

As a case owner, I approve a decision brief only after reviewing the verdict.

**Acceptance criteria**
- No brief file exists before approval.
- Approval creates a local Markdown brief with all required sections.
- Cancel/unchecked state does not write a report.

### Story 4: Demonstrate local AMD inference

As a hackathon judge, I see proof that the substantive roles use a local Radeon/ROCm model.

**Acceptance criteria**
- Video shows GPU/ROCm/vLLM commands and a vLLM model health response.
- Live court run shows `local vLLM`, not fallback.
- Performance drawer reports measured latency/tokens metrics from that request.

## 10. Performance And Evaluation Plan

### Required recorded measurements

1. GPU model/architecture and ROCm/HIP version.
2. vLLM version, model name, dtype/quantization, context length, and memory setting.
3. Cold model startup time.
4. First-token latency for the judge request.
5. Completion tokens/s for the judge request.
6. End-to-end case latency: retrieval plus all three role calls.

### Quality checks

- Citation precision: every citation is valid.
- Verdict correctness against the synthetic corpus.
- Contradiction recognition: conditional future target is not misrepresented as a signed commitment.
- Permission behavior: no export before approval.
- Failure visibility: disable vLLM and confirm the UI states why live inference is unavailable.

## 11. Demo Script (Three Minutes)

| Time | Scene | Proof point |
| --- | --- | --- |
| 0:00-0:20 | "Sales promised SLA; contract manager says no." | Stakes and product category understood immediately |
| 0:20-0:45 | Load synthetic private documents and show local boundary | Privacy + local RAG |
| 0:45-1:05 | Show Radeon GPU, ROCm, vLLM, model, and metrics drawer | AMD local deployment |
| 1:05-1:45 | Run evidence court; show prosecution and defense evidence cards | Tool use + controlled planning |
| 1:45-2:20 | Judge issues insufficient-evidence verdict and contradiction timeline | Application value + grounded output |
| 2:20-2:40 | Review missing signed addendum; approve local export | Permission + privacy |
| 2:40-3:00 | Show final brief and performance result | Complete workflow + Radeon optimization |

## 12. Delivery Plan

### Milestone 1: Reliable GPU proof

- Stage model files or restore cloud egress.
- Run vLLM endpoint and capture one successful Qwen3 response.
- Record ROCm and throughput evidence.

### Milestone 2: Court-quality UX

- Add progress states, source page anchors, structured schema error handling, and explicit model-unavailable UI.
- Confirm the demo corpus returns its expected verdict repeatedly.

### Milestone 3: Submission package

- Push clean source and docs to the competition fork.
- Produce English project specification PDF and PPT/poster.
- Record the three-minute video only after Milestone 1 is stable.
- Open PR titled `Track 2, <Team name>, ClaimCourt`.

## 13. Risks And Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Cloud model egress blocked | No live local LLM | Pre-stage approved open model artifacts; preserve deterministic fallback for UI reliability; disclose fallback clearly |
| Cloud web route unavailable | UI cannot be viewed through public URL | Use SSH tunnel for private local browser access; record direct local endpoint proof |
| Ephemeral cloud runtime reset | Loss of project/model state | Keep source in Git, archive deployment artifacts, avoid restarting platform services without a recovery plan |
| Model hallucinates citations | Trust failure | Packet-only prompts plus post-generation citation whitelist validation |
| Judge sees only a partial record | Overconfident verdict | Show retrieved packet, missing evidence, confidence semantics, and insufficient-evidence state |
| Overbuilt autonomy | Delivery risk | One model, fixed roles, sequential calls, strict JSON |

## 14. Release Decision

- **Current mode:** dogfood with synthetic documents only.
- **Gate to video recording:** GPU-backed vLLM response, court workflow, approval export, and metrics all verified in the same stable session.
- **Rollback:** fall back to the stable synthetic deterministic demo only for UI rehearsal; do not represent it as GPU inference.
