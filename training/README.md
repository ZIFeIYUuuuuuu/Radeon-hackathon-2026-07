# ClaimCourt LoRA Training

This directory builds the `claimcourt-router:8b` adapter. It is not a legal model and it does not replace the Qwen3 32B Q8 judge. Its narrow responsibilities are local task routing, citation-shaped JSON, and secret-safe behavior.

## Data

`generate_claimcourt_sft.py` emits 162 synthetic examples. They cover:

- evidence verdict JSON with citation constraints;
- routing to `evidence_court`, `file_locator`, or `sensitive_record_scan`;
- credential findings that contain only a location, fingerprint, and redacted preview.

The generator contains no real private documents or credentials.

## Train On Radeon Cloud

```bash
python training/generate_claimcourt_sft.py
python training/train_lora.py \
  --model /workspace/models/Qwen3-8B \
  --data training/data/claimcourt_sft.jsonl \
  --output /workspace/claimcourt-models/claimcourt-qwen3-8b-lora
```

## Serve The Adapter

```bash
ollama pull qwen3:8b
ollama create claimcourt-router:8b -f training/Modelfile.claimcourt-router
```

Keep `qwen3:32b-q8_0` as the final evidence judge. The smaller adapter is optimized to choose the right local tool before the court workflow runs.
