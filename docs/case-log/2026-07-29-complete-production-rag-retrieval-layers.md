# Case Log: Complete production RAG retrieval layers

## Intent
Goal: Implement FTS5 persistence, parent-child context retrieval, optional local cross-encoder reranking, and scanned PDF OCR without breaking existing local privacy and memory guarantees.

Success criteria:
- None

Risk: low

## Context
Files read:
- None

SSOT used:
- None

## Gates
First-principles check:
- None

Assumptions:
- None

Adversarial findings:
- None

Edge cases:
- None

Rejected options:
- None

## Explore Card
Mode: explore
Why: Implement FTS5 persistence, parent-child context retrieval, optional local cross-encoder reranking, and scanned PDF OCR without breaking existing local privacy and memory guarantees.
Acceptance:
- {"criterion": "Prototype can be exercised locally", "evidence": "Local smoke check or direct manual exercise", "prohibited": "Do not present a prototype or fixture as production-ready"}
Boundary:
- No production release or irreversible data change


## Changes
- tests/test_core.py
- core.py
- app.py
- README.md
- requirements.txt
- docs/system.md
- scripts/championship_check.py

## Verification
Required evidence level: 0
Achieved evidence level: 1

Commands:
- None

Results:
- python -m unittest discover -s tests -v
- python -m py_compile core.py app.py tests/test_core.py
- python -m unittest discover -s tests -v
- python -m py_compile core.py app.py scripts/championship_check.py tests/test_core.py

## Surprises
- None

## Reusable Lessons
- Durable FTS5 works best as a rebuildable acceleration layer beside the JSON evidence ledger; parent-child context and optional local-only quality models preserve graceful degradation and privacy.

## Follow-ups
- None
