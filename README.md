# ClaimCourt：私有本地 AI 证据法庭

> 面向 AMD AI DevMaster Hackathon 2026 赛道二的冠军目标项目。

ClaimCourt 将本地合同、邮件、会议纪要、聊天记录和演示文稿转换为**带引用的裁决、矛盾时间线和可审批决策简报**。所有文档、检索内容和模型推理都留在 Radeon Cloud 的 AMD Radeon GPU 上，不依赖远程闭源模型 API。

## 项目做什么

- 用户可以用模糊意图找回本地文件，例如“找出我写过的关于客户延期和 Q4 风险的 PPT”。
- 用户可以在本地文件中定位私钥、Token 或凭证，但系统只显示文件位置、类型、指纹和脱敏预览，绝不显示秘密值。
- 对合同、邮件和会议纪要执行检索、检方、辩方、法官三步受控工作流。
- 输出 `supported`、`contradicted` 或 `insufficient_evidence` 裁决，并附证据引用、矛盾、时间线、缺失证据和下一步行动。
- 只有用户明确批准后，系统才会在本地写入 Markdown 决策简报。

## 已验证的 Radeon 配置

- 法官模型：`Qwen3-8B`，vLLM + ROCm，BF16。
- 语义模型：`BAAI/bge-small-zh-v1.5`，vLLM pooling runner，512 维向量。
- 真实门禁结果：23 个文件、28 个文本块、语义检索激活、敏感信息脱敏、真实三角色裁决、禁止 fallback。
- 实测性能：法官首 token 延迟约 `0.23s`，生成速度约 `25.7 tokens/s`。

## RAG 设计

- 先按文档页、PPT 页、段落和句末做结构优先切分，默认约 480 字符、80 字符重叠，为 BGE 的 512-token 上限留出余量。
- embedding 请求按批次发送，超限时自动拆批；单个文件解析失败会记录诊断并跳过，不会阻塞整个工作区。
- 检索同时使用词法、字符 n-gram、语义向量、文件名、文件类型和时间元数据，再按文件族去重。
- JSON 证据账本旁边维护本地 SQLite FTS5 索引；FTS5 只做可重建的检索加速层，不取代可审计的哈希账本。
- 每个父段（文档、页或幻灯片）拆成带 `parent_id`、序号和数量的子块；召回子块后自动合并相邻父上下文，避免只返回半句话。
- 可选加载本地 `sentence-transformers` cross-encoder，对混合召回的候选集做二阶段重排；依赖或模型不可用时保留可解释的混合检索。
- 扫描版 PDF 页面在文字层为空时可调用本地 PyMuPDF + Tesseract OCR，引用会标记为 `page N (OCR)`；未安装 OCR 工具时只跳过该页，不阻塞整个目录。
- 中文查询会识别课程、课设、报告、计算机组成原理和过去一年等意图，降低笔记、试卷、模板对正式报告的干扰。
- 设计参考了 [Pinecone chunking guide](https://www.pinecone.io/learn/chunking-strategies/)、[Azure chunking guidance](https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-chunk-documents)、[LlamaIndex node parsers](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/) 和 [Anthropic Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)。

## 快速体验

```bash
python -m pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8502
```

打开页面后选择 `vLLM ROCm`，模型填写 `Qwen3-8B`，点击“Load stable demo case”，再开启“Championship mode”。冠军演示问题：

> Did the vendor contractually commit to 99.9% uptime?

预期裁决为：没有找到签署的 99.9% SLA 承诺；销售措辞属于营销表达，会议记录是带条件的未来目标，下一步应请求签署 SLA 附录。

## 英文比赛材料

以下英文部分保留给评委、README 审核和官方 PR 使用。

## English Submission Details

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
- **Specialized local router:** an optional Qwen3 8B LoRA adapter is trained locally on synthetic ClaimCourt routing, cited-verdict, and secret-redaction examples. It selects local tools; the verified local Qwen3-8B vLLM service is the final judge.
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
      --judge-model Qwen3-8B \
      --require-live-judge

The app includes a deterministic local fallback so the complete UI and demo corpus can be reviewed before the GPU server is ready. For the competition demo, start the local ROCm service below; the court record then reports the active local runtime, judge first-token latency, and generated token throughput.

## Radeon Cloud / ROCm deployment

Select **ROCm vLLM-dev (Navi)** in Radeon Cloud. First validate the instance:

```bash
rocminfo
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
vllm --version
```

For the W7900-class 48 GB instance, the verified competition configuration is Qwen3-8B in vLLM. The BF16 weights use about 15.34 GiB and leave about 20 GiB for KV cache, making the three-call court workflow stable on the Radeon instance. Ollama remains an optional compatibility runtime:

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

ClaimCourt also maintains a durable SQLite FTS5 index beside the JSON ledger. The
index is rebuilt from the hashed evidence after every ingestion, so it can be
deleted and regenerated without losing the audit history. Retrieval first uses
hybrid lexical/embedding scores, then optionally reranks a shortlist with a local
`sentence-transformers` cross-encoder. To enable that path, install the optional
package and point the UI at a model already present in the private model cache:

```bash
python -m pip install sentence-transformers
# optional, for scanned PDFs:
python -m pip install pymupdf pytesseract pillow
# the tesseract executable and the selected language packs must also be local
```

Every retrieved child chunk carries its structural parent and neighboring context
in the evidence packet. This keeps citations anchored to a small chunk while
giving the judge enough surrounding text to distinguish an exception clause from
the sentence immediately before it.

The verified vLLM judge is served with:

```bash
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
vllm serve /workspace/models/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.78 \
  --max-model-len 8192 \
  --served-model-name Qwen3-8B \
  --enforce-eager
```

Start the interface, select **vLLM ROCm**, and use `http://localhost:8000/v1` with model `Qwen3-8B`. The app sends `enable_thinking=false` for the court calls so the structured JSON contract is not polluted by a reasoning stream.

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
