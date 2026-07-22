# ClaimCourt Recovery Runbook

The source of truth is the `claimcourt` branch of the fork:

https://github.com/ZIFeIYUuuuuuu/Radeon-hackathon-2026-07/tree/claimcourt

The local machine backup is under `backups/cloud-2026-07-22/`. It contains the trained LoRA adapter, its metrics, and the persistent workspace index. Do not commit that directory; the adapter is intentionally kept outside GitHub because it is about 175 MB.

## New Radeon Cloud Instance

Use a ROCm image, then run:

```bash
git clone --depth 1 --branch claimcourt https://github.com/ZIFeIYUuuuuuu/Radeon-hackathon-2026-07.git /workspace/claimcourt
cd /workspace/claimcourt
/opt/venv/bin/python -m pip install -r requirements.txt
```

Start the local Ollama runtime using the ROCm runtime available in the instance, then restore the main model:

```bash
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS=/workspace/ollama-models
ollama serve > /workspace/ollama.log 2>&1 &
ollama pull qwen3:32b-q8_0
```

Restore the adapter and index from the local backup when needed:

```bash
mkdir -p /workspace/claimcourt-models/claimcourt-qwen3-8b-lora /workspace/claimcourt/data
# Transfer backups/cloud-2026-07-22/claimcourt-qwen3-8b-lora here.
# Transfer backups/cloud-2026-07-22/workspace_index.json to /workspace/claimcourt/data/.
```

Start the UI:

```bash
cd /workspace/claimcourt
nohup /opt/venv/bin/streamlit run app.py --server.address 127.0.0.1 --server.port 8502 > /workspace/claimcourt-ui.log 2>&1 &
```

Verify recovery:

```bash
curl http://127.0.0.1:11434/api/version
ollama ps
rocm-smi --showmeminfo vram
PYTHONPATH=. /opt/venv/bin/python -m unittest discover -s tests -v
```

The 32B model can be re-downloaded from the Ollama registry. The source code, synthetic training corpus, training script, adapter backup, and workspace history are the irreplaceable project state.
