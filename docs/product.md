# Product SSOT

## Product
- Name: ClaimCourt: A Private Local AI Evidence Court
- Category: Private AI agent for evidence-grounded business decisions
- Promise: Turn a disputed claim across private documents into a cited verdict, contradiction timeline, and approval-gated decision brief without exporting the evidence to a closed remote model.

## Target Users
- Primary user: An operations, procurement, customer-success, or legal-operations lead preparing for a dispute or negotiation.
- Buyer, if different: A security-conscious team lead responsible for private AI deployment.
- Jobs to be done:
  - Determine what the record actually supports before escalating a dispute.
  - Separate binding commitments from sales language, drafts, exceptions, and assumptions.
  - Produce an auditable brief another human can inspect and approve.

## Scope
### In Scope
- Local intake of PDF, DOCX, TXT, Markdown, and EML evidence.
- Local chunking and retrieval with stable citation identifiers.
- A controlled prosecution -> defense -> judge workflow using one local instruction model.
- Structured verdict JSON: claim, verdict, confidence, citations, contradictions, timeline, missing evidence, and next action.
- Citation whitelist enforcement and explicit insufficient-evidence outcomes.
- Local session memory, approval-gated Markdown export, and Radeon/ROCm performance telemetry.
- Synthetic SLA dispute demo corpus and a deterministic fallback only when local inference is unavailable.

### Out of Scope
- Legal advice, autonomous negotiation, automatic contract modification, or a claim that the verdict is legally binding.
- Training or fine-tuning a model during the hackathon.
- Multi-model autonomous debate, browser automation, external closed-model APIs, or public sharing of real private documents.

## Core Workflows
1. Load the synthetic demo or upload trusted local evidence.
2. State a disputed claim.
3. Retrieve an evidence packet with stable citations.
4. Run prosecution, defense, and judge roles against that packet only.
5. Inspect verdict, evidence, contradictions, timeline, and missing evidence.
6. Explicitly approve export of a local decision brief.

## Roadmap
### Now
- Reliable Radeon Cloud deployment, model artifact availability, vLLM inference, deterministic demo, and video-ready telemetry.

### Next
- Persistent encrypted local case memory, hybrid embeddings, source-page anchors, role-level progress, and a benchmark panel.

### Later
- Team roles, signed case audit logs, policy packs, private on-premises deployment, and evaluation datasets across procurement and approval disputes.

## Success Metrics
- Every displayed verdict cites only retrieved evidence.
- The SLA demo reaches the intended `insufficient_evidence` verdict with a signed-addendum next action.
- A first-time viewer can explain the product in 10 seconds: "AI puts private evidence on trial locally."
- Radeon Cloud demo records model name, quantization/dtype, first-token latency, end-to-end latency, and output tokens/s.
- No real document or model prompt reaches a closed remote API.
