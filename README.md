# ClaimCourt: A Private Local AI Evidence Court

ClaimCourt turns private contracts, emails, meeting notes, and chat exports into a cited decision brief. Its controlled workflow retrieves local evidence, presents the strongest case on each side, reconstructs contradictions and a timeline, then produces a verdict that can cite only the retrieved packet.

It is designed for **AMD AI DevMaster Hackathon 2026, Track 2**. No remote closed model API is used. The model endpoint is a local Ollama ROCm or vLLM server running on an AMD Radeon GPU.

## What it demonstrates

- **Local workspace retrieval:** recursive local indexing of PDF, DOCX, PPTX, TXT, Markdown, and EML files; PDF pages and PPTX slides retain local locators.
- **Controlled tool workflow:** prosecutor, defense, and judge steps are separate constrained calls.
- **Multi-step planning:** retrieve -> prosecution -> defense -> judge -> approval-gated export.
- **Local memory:** the indexed evidence and court record remain in the current session on the host.
- **Privacy and permissions:** no document leaves the machine; a local decision brief is written only after the user checks the approval control.
- **Citation guardrail:** the judge may cite only IDs supplied in the retrieved evidence packet. Unknown citations are removed before rendering or export.
- **Evidence ledger:** every rendered source and excerpt carries a local SHA-256 hash and chunk locator; approved briefs include this ledger for later verification.
- **Secret-location tool:** deterministic local scanning finds likely private keys and credentials, but returns only a file location, category, fingerprint, and redacted preview.
- **Runtime proof:** the UI can query the local Ollama or vLLM endpoint and display the active runtime, GPU residency, VRAM, and context facts.
- **Specialized local router:** an optional Qwen3 8B LoRA adapter is trained locally on synthetic ClaimCourt routing, cited-verdict, and secret-redaction examples. It selects local tools; the Qwen3 32B Q8 model remains the final judge.
- **Unified private request:** one request routes to evidence court, file location, or a redacted sensitive-record scan without sending workspace contents outside the instance.
- **Semantic memory finder:** vague requests such as “find the PPT I wrote about customer delay and Q4 risk” are converted into a transparent local intent plan, expanded concepts, file-type constraints, hybrid relevance scores, and human-readable match reasons.
- **Optional local embeddings:** when a loopback Ollama /api/embed or vLLM /v1/embeddings endpoint is configured, embedding similarity is blended with lexical retrieval; an unavailable endpoint falls back without sending data elsewhere.
- **Query memory:** local query plans, result identities, confidence, and the embedding model used are persisted with the workspace index for replay and audit.
- **File-family grouping:** similar copies and version-like names are grouped so a PDF export, draft, and final presentation do not appear as unrelated discoveries.
- **Accumulating workspace memory:** the local index persists source hashes, current evidence, and prior-version excerpts across refreshes; changed files are marked and re-indexed instead of silently reusing stale evidence.

## Architecture

```text
Local PDF / DOCX / TXT / MD / EML
          |
  parse + chunk + local retrieval
          |
  retrieved evidence packet with citation IDs
          |
  prosecution -> defense -> judge (same local instruction model)
          |
cited verdict + contradictions + timeline + missing evidence
          |
local evidence ledger + redacted sensitive-record locator
          |
explicit user approval -> local Markdown decision brief
```

## Quick demo

```bash
cd claimcourt
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Click **Load stable demo case**, then ask:

> Did the vendor contractually commit to 99.9% uptime?

The included corpus intentionally produces `Insufficient Evidence`: sales language is marketing, meeting notes describe a conditional future target, and the agreement requires a signed SLA addendum.

For the championship acceptance run, build the larger multi-file workspace and run its gate:

    python scripts/build_championship_corpus.py
    python scripts/championship_check.py --output championship_results.json

The live release gate additionally requires the local embedding and judge endpoints:

    python scripts/championship_check.py \
      --embedding-endpoint http://localhost:8001/v1 \
      --embedding-model /workspace/models/bge-small-zh-v1.5 \
      --judge-endpoint http://localhost:8000/v1 \
      --judge-model Qwen/Qwen3-8B \
      --require-live-judge

The app includes a deterministic local fallback so the complete UI and demo corpus can be reviewed before the GPU server is ready. For the competition demo, start the local ROCm service below; the court record then reports the active local runtime, judge first-token latency, and generated token throughput.

## Radeon Cloud / ROCm deployment

Select **ROCm vLLM-dev (Navi)** in Radeon Cloud. First validate the instance:

```bash
rocminfo
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
vllm --version
```

For the W7900-class 48 GB instance, the recommended competition configuration is `qwen3:32b-q8_0`. Its 35.1 GB Q8 weights prioritize document reasoning quality while leaving headroom for a 16K evidence context. Run it through a local ROCm Ollama server:

```bash
export PATH=/workspace/ollama-runtime/bin:$PATH
export LD_LIBRARY_PATH=/workspace/ollama-runtime/lib:$LD_LIBRARY_PATH
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS=/workspace/ollama-models
ollama serve
```

In a second terminal, pull and verify the model:

```bash
ollama pull qwen3:32b-q8_0
ollama run qwen3:32b-q8_0 "Return JSON only: {\"status\": \"ready\"}"
ollama ps
rocm-smi
```

For semantic retrieval, run a local embedding model in Ollama and enable **Use local semantic embeddings** in the sidebar:

    ollama pull nomic-embed-text

The embedding endpoint is http://localhost:11434 with model nomic-embed-text. This is optional; the app keeps a deterministic hybrid fallback when the embedding model is not loaded.

On the Radeon Cloud deployment, the verified configuration uses vLLM's pooling runner:

    vllm serve /workspace/models/bge-small-zh-v1.5 --runner pooling --host 0.0.0.0 --port 8001 --dtype bfloat16 --gpu-memory-utilization 0.12 --max-model-len 512

Use http://localhost:8001/v1 and model /workspace/models/bge-small-zh-v1.5 in ClaimCourt. The endpoint exposes /v1/embeddings and returns 512-dimensional vectors.

Start the interface, select **Ollama ROCm**, and use `http://localhost:11434` with model `qwen3:32b-q8_0`. The app uses Ollama's native streaming telemetry for first-token latency and output throughput.

vLLM remains a supported fallback for the preinstalled Qwen3 8B model:

```bash
vllm serve Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

For constrained VRAM, prefer a supported AWQ/GPTQ quantized Qwen3 checkpoint and pass its documented `--quantization` option. Measure and record first-token latency and output tokens/second on the app's court request. This is evidence for the AMD/ROCm performance scoring section.

Run the UI in a second terminal:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Set the sidebar endpoint to `http://localhost:8000/v1` (or the private instance address), select **vLLM ROCm**, and enable local GPU inference. ClaimCourt intentionally does not accept a cloud-model API key.

## Optional ClaimCourt LoRA

The `training/` directory contains a reproducible local LoRA specialization path for `Qwen3-8B`. It is deliberately narrow: route local requests to the court, file locator, or redacted secret scanner. It does not replace the larger evidence judge.

```bash
python training/generate_claimcourt_sft.py
python training/train_lora.py \
  --model /workspace/models/Qwen3-8B \
  --data training/data/claimcourt_sft.jsonl \
  --output /workspace/claimcourt-models/claimcourt-qwen3-8b-lora
```

See `training/README.md` for adapter serving instructions and dataset scope. Workspace evidence history and prior-version archive are stored locally under `data/workspace_index.json`.

## Tests

```bash
cd claimcourt
python -m unittest discover -s tests -v
```

## Privacy model

Documents are parsed and indexed in process. Retrieval data, prompts, structured arguments, and reports remain on the current host. The interface only permits a report file write after explicit approval. Do not use the demo interface to ingest data into a server outside your trusted private network.

## Submission checklist

- Fork `AMD-DEV-CONTEST/Radeon-hackathon-2026-07` and add this project.
- Use PR title: `Track 2, <Team name>, ClaimCourt`.
- Submit all materials in English: source, this README, a project specification PDF, demo video, and a poster or PPT.
- In the video, show `rocminfo`, the local ROCm runtime, the Q8 model resident on GPU, the live GPU-backed verdict, and generated telemetry.
