"""Run the no-fallback acceptance gate for the ClaimCourt championship corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import LocalEmbeddingClient, LocalRetriever, LocalVLLM, discover_workspace, run_court


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path(__file__).parent.parent / "championship_corpus")
    parser.add_argument("--embedding-endpoint", default="")
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--judge-endpoint", default="")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--require-live-judge", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("championship_results.json"))
    args = parser.parse_args()

    embedder = None
    if args.embedding_endpoint and args.embedding_model:
        embedder = LocalEmbeddingClient("vLLM ROCm", args.embedding_endpoint, args.embedding_model)
    retriever = LocalRetriever(embedder=embedder)
    files = discover_workspace(args.corpus)
    chunks = retriever.index_paths(files)

    memory_matches = retriever.locate_files(
        "I wrote a presentation about the customer's delayed launch and Q4 risk; find it"
    )
    if not memory_matches or memory_matches[0].source != "Customer_Delivery_Risk_Q4_Final.pptx":
        raise AssertionError("Memory Finder did not rank the final presentation first.")

    findings = retriever.scan_sensitive_records()
    if not findings or any("FAKE_DEMO_KEY_MATERIAL" in item.redacted_preview for item in findings):
        raise AssertionError("Sensitive locator did not return a redacted finding.")

    evidence = retriever.search("Did the vendor contractually commit to 99.9% uptime?", limit=8)
    court_mode = "not_run"
    verdict = None
    telemetry = {}
    if args.judge_endpoint and args.judge_model:
        _, _, verdict, telemetry, court_mode = run_court(
            "Did the vendor contractually commit to 99.9% uptime?",
            evidence,
            LocalVLLM(args.judge_endpoint, args.judge_model),
        )
        if args.require_live_judge and court_mode == "local rule fallback":
            raise AssertionError("Judge endpoint failed; championship mode forbids fallback.")
        if verdict.get("verdict") != "insufficient_evidence":
            raise AssertionError(f"Unexpected championship verdict: {verdict.get('verdict')}")
    elif args.require_live_judge:
        raise AssertionError("--require-live-judge needs --judge-endpoint and --judge-model.")

    result = {
        "corpus_files": len(files),
        "indexed_chunks": chunks,
        "embedding_model": retriever.embedding_model or None,
        "embedding_active": retriever.embedding_matrix is not None,
        "memory_finder": {
            "source": memory_matches[0].source,
            "score": memory_matches[0].score,
            "confidence": memory_matches[0].confidence,
            "reasons": list(memory_matches[0].reasons),
            "grouped_similar_versions": len(memory_matches[0].duplicate_paths),
        },
        "sensitive_locator": {
            "count": len(findings),
            "redacted": all("FAKE_DEMO_KEY_MATERIAL" not in item.redacted_preview for item in findings),
        },
        "court": {"mode": court_mode, "verdict": verdict, "telemetry": telemetry},
    }
    args.output.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
