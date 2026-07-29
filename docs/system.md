# System SSOT

## Architecture
```text
Trusted local documents
  -> parser and chunker
  -> durable JSON evidence ledger + SQLite FTS5, hybrid child retrieval and parent context
  -> controlled evidence packet
  -> prosecution -> defense -> judge (one local vLLM model)
  -> citation validation, timeline, and verdict rendering
  -> explicit approval -> local Markdown brief
```

## Tech Stack
- Frontend: Streamlit.
- Backend: Python 3.10+, requests, structured JSON workflow.
- Data: JSON evidence ledger with source hashes, adjacent SQLite FTS5 acceleration, in-memory TF-IDF/BM25/embedding matrices; source files remain local to the deployment boundary.
- Inference: OpenAI-compatible local vLLM endpoint on AMD Radeon GPU + ROCm; Qwen3-8B is the primary target.
- Infrastructure: Radeon Cloud `ROCm vLLM-dev (Navi)` image; SSH tunnel is the reliable private browser access path when cloud web routing fails.

## Important Commands
```bash
# install
cd claimcourt
python -m pip install -r requirements.txt

# test
PYTHONPATH=. python tests/test_core.py

# local vLLM target on Radeon Cloud
vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 8000 --dtype bfloat16 --gpu-memory-utilization 0.85 --max-model-len 8192

# UI
streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

## Data Model
- `Evidence`: citation, source, text, relevance score, optional date, structural parent ID, child position, and optional merged parent context.
- `Case`: user claim, retrieved packet, role outputs, verdict, telemetry, and export approval state.
- `Verdict`: supported, contradicted, or insufficient_evidence; confidence; reasoning; citations; contradictions; timeline; missing evidence; recommended action.

## Integrations
- Local OpenAI-compatible vLLM API only. No closed remote model API is permitted.
- Radeon Cloud is an execution environment, not a data source.

## Deployment
- Primary: Radeon Cloud GPU inference with a local browser connection through SSH port forwarding.
- Browser route: `ssh -L 8502:127.0.0.1:8502 ...`, then open `http://localhost:8502`.
- Fallback: deterministic, visibly labeled demo mode when the local LLM endpoint is unavailable. It is for reliability, not a substitute for GPU inference scoring.

## Guardrails
- Auth/security: Require a trusted private deployment; do not expose the UI publicly with real evidence.
- Data safety: Parse/index in process; only export after explicit approval; reject citations absent from the packet.
- Cost/latency: Capture first-token latency and judge output tokens/s; preserve model cache; avoid instance restarts.
- Optional quality layers: local cross-encoder reranking and PyMuPDF/Tesseract OCR are lazy and fail closed to the deterministic hybrid index.
- Rollback: Source and demo corpus live in Git; archive before cloud changes; stop only app processes, never destroy a working instance without preserving evidence.
