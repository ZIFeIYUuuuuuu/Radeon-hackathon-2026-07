from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from core import (
    FileMatch,
    LocalEmbeddingClient,
    LocalOllama,
    LocalRetriever,
    LocalVLLM,
    discover_workspace,
    local_runtime_status,
    markdown_brief,
    route_workspace_request,
    run_court,
)


ROOT = Path(__file__).parent
DEMO_DIR = ROOT / "demo_corpus"
UPLOAD_DIR = ROOT / "data" / "uploads"
EXPORT_DIR = ROOT / "data" / "exports"
INDEX_FILE = ROOT / "data" / "workspace_index.json"

st.set_page_config(page_title="ClaimCourt", page_icon="⚖", layout="wide")

if "retriever" not in st.session_state:
    st.session_state.retriever = LocalRetriever(INDEX_FILE)
if "case" not in st.session_state:
    st.session_state.case = None
if "runtime_status" not in st.session_state:
    st.session_state.runtime_status = None
if "sensitive_findings" not in st.session_state:
    st.session_state.sensitive_findings = []
if "file_matches" not in st.session_state:
    st.session_state.file_matches = []
if "intent_plan" not in st.session_state:
    st.session_state.intent_plan = None


def index_demo() -> None:
    embedder = st.session_state.retriever.embedder
    st.session_state.retriever = LocalRetriever(INDEX_FILE, embedder=embedder)
    count = st.session_state.retriever.index_paths(DEMO_DIR.glob("*"))
    st.session_state.indexed_count = count
    st.session_state.sensitive_findings = []
    st.session_state.file_matches = []
    st.session_state.intent_plan = None


st.title("ClaimCourt")
st.caption("Private local AI evidence court | documents stay on this machine")

with st.sidebar:
    st.subheader("Local Inference")
    runtime = st.selectbox("Runtime", ["Ollama ROCm", "vLLM ROCm"])
    default_endpoint = "http://localhost:11434" if runtime == "Ollama ROCm" else "http://localhost:8000/v1"
    default_model = "qwen3:32b-q8_0" if runtime == "Ollama ROCm" else "Qwen/Qwen3-8B"
    endpoint = st.text_input("Local runtime endpoint", default_endpoint)
    model = st.text_input("Local model", default_model)
    use_local_inference = st.toggle("Use local GPU inference", value=True)
    st.caption("No external model provider is contacted. The selected runtime must run on this Radeon Cloud instance.")
    if st.button("Check local runtime", use_container_width=True):
        st.session_state.runtime_status = local_runtime_status(runtime, endpoint, model)
    status = st.session_state.runtime_status
    if status:
        if status.get("available"):
            st.success(f"{status['service']} | {status['processor']}")
            if status.get("model_loaded"):
                vram_gib = status.get("size_vram", 0) / (1024 ** 3)
                context = status.get("context_length", 0)
                st.caption(f"Model resident locally | VRAM: {vram_gib:.1f} GiB | Context: {context or 'unknown'}")
            else:
                st.warning("Runtime is reachable, but the selected model is not loaded.")
        else:
            st.error(f"Local runtime unavailable: {status.get('error', 'unknown error')}")
    st.divider()
    st.subheader("Semantic Retrieval")
    use_embeddings = st.toggle("Use local semantic embeddings", value=False)
    if use_embeddings:
        embedding_runtime = st.selectbox("Embedding runtime", ["Ollama ROCm", "vLLM ROCm"])
        embedding_endpoint = st.text_input(
            "Embedding endpoint",
            os.getenv(
                "CLAIMCOURT_EMBEDDING_ENDPOINT",
                "http://localhost:11434" if embedding_runtime == "Ollama ROCm" else "http://localhost:8001/v1",
            ),
        )
        embedding_model = st.text_input(
            "Embedding model",
            os.getenv(
                "CLAIMCOURT_EMBEDDING_MODEL",
                "nomic-embed-text" if embedding_runtime == "Ollama ROCm" else "/workspace/models/bge-small-zh-v1.5",
            ),
        )
        st.session_state.retriever.embedder = LocalEmbeddingClient(
            embedding_runtime,
            embedding_endpoint,
            embedding_model,
        )
        st.caption("Embeddings are optional. If the local endpoint is unavailable, deterministic hybrid retrieval remains active.")
    else:
        st.session_state.retriever.embedder = None
        st.session_state.retriever.embedding_model = ""
    st.divider()
    st.subheader("Evidence Intake")
    workspace = st.text_input("Local workspace folder", placeholder="/workspace/private-case")
    uploads = st.file_uploader("Add private documents", type=["pdf", "docx", "pptx", "txt", "md", "eml"], accept_multiple_files=True)
    if st.button("Load stable demo case", use_container_width=True):
        index_demo()
        st.success(f"Indexed {st.session_state.indexed_count} local evidence chunks.")
    if uploads and st.button("Index selected documents", use_container_width=True):
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        saved = []
        for upload in uploads:
            destination = UPLOAD_DIR / upload.name
            destination.write_bytes(upload.getbuffer())
            saved.append(destination)
        embedder = st.session_state.retriever.embedder
        st.session_state.retriever = LocalRetriever(INDEX_FILE, embedder=embedder)
        st.session_state.indexed_count = st.session_state.retriever.index_paths(saved, accumulate=True)
        st.session_state.sensitive_findings = []
        st.session_state.file_matches = []
        st.session_state.intent_plan = None
        st.success(f"Indexed {st.session_state.indexed_count} local evidence chunks.")
    if st.button("Index local workspace", use_container_width=True):
        try:
            files = discover_workspace(Path(workspace).expanduser())
            embedder = st.session_state.retriever.embedder
            st.session_state.retriever = LocalRetriever(INDEX_FILE, embedder=embedder)
            st.session_state.indexed_count = st.session_state.retriever.index_paths(files, accumulate=True)
            st.session_state.sensitive_findings = []
            st.session_state.file_matches = []
            st.session_state.intent_plan = None
            st.success(f"Indexed {len(files)} local files into {st.session_state.indexed_count} evidence chunks.")
        except ValueError as exc:
            st.error(str(exc))
    if st.button("Scan indexed files for sensitive records", use_container_width=True, disabled=not st.session_state.retriever.evidence):
        st.session_state.sensitive_findings = st.session_state.retriever.scan_sensitive_records()

left, right = st.columns([1.2, 1])
with left:
    claim = st.text_area(
        "Private workspace request",
        "Did the vendor contractually commit to 99.9% uptime?",
        height=90,
    )
with right:
    st.markdown("#### Privacy boundary")
    st.write("Retrieval, agent calls, memory, and export run locally. Export requires an explicit approval below.")
    st.write(f"Indexed chunks: **{len(st.session_state.retriever.evidence)}**")
    if st.session_state.retriever.embedding_model:
        st.caption(f"Semantic embeddings: {st.session_state.retriever.embedding_model}")
    elif st.session_state.retriever.embedder:
        st.caption("Semantic embeddings configured; they will activate on the next index.")
    if st.session_state.retriever.history:
        st.caption(f"Persistent history: {len(st.session_state.retriever.history)} version event(s)")
    if st.session_state.retriever.query_history:
        with st.expander(f"Query memory ({len(st.session_state.retriever.query_history)})"):
            for item in reversed(st.session_state.retriever.query_history[-10:]):
                st.caption(f"{item['query_id']} · {item['created_at']}")
                st.write(item["query"])
                st.caption(f"Intent: {item['intent']['intent']} · results: {len(item['results'])}")

if st.button("Run private workspace request", type="primary", use_container_width=True):
    try:
        route = route_workspace_request(claim)
        st.session_state.file_matches = []
        if route == "sensitive_record_scan":
            st.session_state.sensitive_findings = st.session_state.retriever.scan_sensitive_records()
            st.session_state.case = None
            st.success("Completed local redacted sensitive-record scan.")
        elif route == "file_locator":
            st.session_state.file_matches = st.session_state.retriever.locate_files(claim)
            st.session_state.intent_plan = st.session_state.retriever.last_intent
            st.session_state.sensitive_findings = []
            st.session_state.case = None
            st.success("Completed local file-location search.")
        else:
            retrieved = st.session_state.retriever.search(claim)
            if runtime == "Ollama ROCm":
                llm = LocalOllama(endpoint, model) if use_local_inference else None
            else:
                llm = LocalVLLM(endpoint, model) if use_local_inference else None
            prosecution, defense, verdict, telemetry, mode = run_court(claim, retrieved, llm)
            st.session_state.case = {"retrieved": retrieved, "prosecution": prosecution, "defense": defense, "verdict": verdict, "telemetry": telemetry, "mode": mode}
    except ValueError as exc:
        st.error(str(exc))

findings = st.session_state.sensitive_findings
if findings:
    st.warning(f"{len(findings)} sensitive record(s) located locally. Values are never displayed or exported by this view.")
    st.dataframe(
        [
            {
                "Source": item.source,
                "Location": item.locator,
                "Type": item.kind,
                "Fingerprint": item.fingerprint,
                "Preview": item.redacted_preview,
            }
            for item in findings
        ],
        use_container_width=True,
        hide_index=True,
    )

if st.session_state.retriever.history:
    with st.expander("Workspace version history"):
        st.caption("A changed source is recorded by hash and re-indexed as a new current version.")
        st.dataframe(st.session_state.retriever.history[-20:], use_container_width=True, hide_index=True)
        archived = st.session_state.retriever.archived_versions()
        if archived:
            st.caption(f"Archived prior evidence versions: {len(archived)} excerpt(s)")
            st.dataframe(
                [
                    {
                        "Source": item.source,
                        "Location": item.locator,
                        "Previous SHA-256": item.source_sha256[:16],
                        "Previous excerpt": item.text[:180],
                    }
                    for item in archived[-20:]
                ],
                use_container_width=True,
                hide_index=True,
            )

matches = st.session_state.file_matches
if matches:
    st.subheader("Local file matches")
    plan = st.session_state.intent_plan
    if plan:
        st.caption(
            f"Intent: **{plan.intent}** · types: {', '.join(plan.artifact_types)} · "
            f"topics: {', '.join(plan.topics) or 'general'} · confidence: {plan.confidence:.0%}"
        )
        with st.expander("Inspect local search plan"):
            st.json({
                "intent": plan.intent,
                "artifact_types": list(plan.artifact_types),
                "topics": list(plan.topics),
                "entities": list(plan.entities),
                "time_hints": list(plan.time_hints),
                "expanded_terms": list(plan.expanded_terms),
                "search_scope": list(plan.search_scope),
                "confidence": plan.confidence,
            })
    for match in matches:
        if isinstance(match, FileMatch):
            with st.container(border=True):
                st.markdown(f"**{match.source}** · confidence {match.confidence:.0%} · score {match.score:.3f}")
                st.caption(f"File family: {match.file_family}")
                st.markdown("**Why this matched**")
                for reason in match.reasons:
                    st.write(f"- {reason}")
                if match.duplicate_paths:
                    st.caption(f"Grouped similar copies/versions: {len(match.duplicate_paths)}")
                for item in match.evidence:
                    with st.expander(f"[{item.citation}] {item.locator}"):
                        st.write(item.text)
        else:
            st.dataframe(
                [{
                    "Source": item.source,
                    "Location": item.locator,
                    "Relevance": item.score,
                    "Excerpt": item.text[:220],
                } for item in matches],
                use_container_width=True,
                hide_index=True,
            )

case = st.session_state.case
if case:
    verdict = case["verdict"]
    st.divider()
    st.subheader("Court Record")
    metrics = st.columns(5)
    metrics[0].metric("Verdict", verdict["verdict"].replace("_", " ").title())
    metrics[1].metric("Confidence", f"{float(verdict.get('confidence', 0)):.0%}")
    metrics[2].metric("Inference", case["mode"])
    metrics[3].metric("Judge first token", f"{case['telemetry'].get('first_token_latency_seconds', '-')} s")
    metrics[4].metric("Judge tokens/s", case["telemetry"].get("tokens_per_second", "-"))
    st.write(verdict.get("reasoning", ""))

    prosecution_col, defense_col = st.columns(2)
    with prosecution_col:
        st.markdown("#### Prosecution")
        st.write(case["prosecution"].get("position", ""))
        st.caption("Citations: " + ", ".join(case["prosecution"].get("citations", [])))
    with defense_col:
        st.markdown("#### Defense")
        st.write(case["defense"].get("position", ""))
        st.caption("Citations: " + ", ".join(case["defense"].get("citations", [])))

    st.markdown("#### Cited evidence")
    by_citation = {item.citation: item for item in case["retrieved"]}
    for citation in verdict.get("evidence_citations", []):
        item = by_citation.get(citation)
        if item:
            with st.expander(f"[{item.citation}] {item.source} | relevance {item.score}"):
                st.write(item.text)

    st.markdown("#### Evidence ledger")
    st.caption("Each source and displayed excerpt is hashed locally so a reviewer can verify the record used for this verdict.")
    st.dataframe(
        [
            {
                "Citation": item.citation,
                "Source": item.source,
                "Location": item.locator,
                "Source SHA-256": item.source_sha256[:16],
                "Excerpt SHA-256": item.evidence_sha256[:16],
            }
            for item in case["retrieved"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    timeline = verdict.get("timeline_events", [])
    if timeline:
        st.markdown("#### Contradiction timeline")
        for event in timeline:
            st.write(f"**{event.get('date', 'Undated')}**  [{event.get('citation', '')}] {event.get('event', '')}")
    st.markdown("#### Missing evidence")
    for item in verdict.get("missing_evidence", []):
        st.write(f"- {item}")
    st.markdown("#### Recommended next action")
    st.info(verdict.get("recommended_next_action", ""))

    st.divider()
    approved = st.checkbox("I approve writing this decision brief to a local file.")
    if approved:
        brief = markdown_brief(verdict, case["retrieved"])
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        export_path = EXPORT_DIR / "claimcourt-decision-brief.md"
        export_path.write_text(brief, encoding="utf-8")
        st.success(f"Local brief written to {export_path}")
        st.download_button("Download approved brief", brief, file_name=export_path.name, mime="text/markdown")
