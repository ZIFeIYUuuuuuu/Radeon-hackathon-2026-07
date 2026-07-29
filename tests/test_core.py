from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from core import BM25Index, LocalEmbeddingClient, LocalRetriever, _document_sections, discover_workspace, infer_intent, markdown_brief, route_workspace_request, run_court, scan_sensitive_paths


class ClaimCourtTests(unittest.TestCase):
    def setUp(self):
        self.retriever = LocalRetriever()
        self.retriever.index_paths((Path(__file__).parent.parent / "demo_corpus").glob("*"))

    def test_demo_case_returns_cited_verdict(self):
        evidence = self.retriever.search("Did the vendor contractually commit to 99.9% uptime?")
        _, _, verdict, _, _ = run_court("Did the vendor contractually commit to 99.9% uptime?", evidence)
        self.assertEqual("insufficient_evidence", verdict["verdict"])
        self.assertTrue(verdict["evidence_citations"])

    def test_short_demo_corpus_does_not_repeat_terminal_chunks(self):
        self.assertGreaterEqual(len(self.retriever.evidence), 5)
        self.assertTrue(all(len(item.text) <= 480 for item in self.retriever.evidence))

    def test_unreadable_files_do_not_abort_directory_indexing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            readable = root / "readable.md"
            unreadable = root / "empty.docx"
            readable.write_text("a readable local report", encoding="utf-8")
            unreadable.write_bytes(b"")
            retriever = LocalRetriever()
            retriever.index_paths([readable, unreadable])
        self.assertEqual(["readable.md"], [item.source for item in retriever.evidence])
        self.assertEqual(1, len(retriever.ingestion_errors))
        self.assertEqual("unreadable", retriever.ingestion_errors[0]["status"])

    def test_brief_contains_local_evidence(self):
        evidence = self.retriever.search("99.9% uptime")
        _, _, verdict, _, _ = run_court("Did the vendor contractually commit to 99.9% uptime?", evidence)
        brief = markdown_brief(verdict, evidence)
        self.assertIn("Cited Evidence", brief)
        self.assertIn("Evidence Ledger", brief)

    def test_evidence_ledger_has_stable_hashes(self):
        entry = self.retriever.evidence[0]
        ledger = self.retriever.evidence_ledger([entry])
        self.assertEqual(entry.citation, ledger[0]["citation"])
        self.assertEqual(64, len(ledger[0]["source_sha256"]))
        self.assertEqual(64, len(ledger[0]["evidence_sha256"]))

    def test_sensitive_scan_redacts_private_material(self):
        fake_token = "ghp_" + "A" * 36
        with TemporaryDirectory() as directory:
            path = Path(directory) / "notes.txt"
            path.write_text(f"temporary token = {fake_token}\n", encoding="utf-8")
            findings = scan_sensitive_paths([path])
        self.assertEqual(1, len(findings))
        self.assertEqual("github token", findings[0].kind)
        self.assertNotIn(fake_token, findings[0].redacted_preview)
        self.assertTrue(findings[0].fingerprint.startswith("sha256:"))

    def test_workspace_discovery_and_request_routing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            (nested / "proposal.pptx").write_bytes(b"placeholder")
            (root / "notes.md").write_text("local note", encoding="utf-8")
            (root / "ignore.exe").write_bytes(b"not evidence")
            self.assertEqual(["notes.md", "proposal.pptx"], sorted(path.name for path in discover_workspace(root)))
        self.assertEqual("file_locator", route_workspace_request("Find the renewal presentation deck"))
        self.assertEqual("sensitive_record_scan", route_workspace_request("Where is my private key?"))
        self.assertEqual("evidence_court", route_workspace_request("Did the vendor approve the delay?"))

    def test_vague_request_becomes_transparent_intent_plan(self):
        plan = infer_intent("I wrote a PPT about customer delay and Q4 delivery risk, find it")
        self.assertEqual("locate_artifact", plan.intent)
        self.assertIn(".pptx", plan.artifact_types)
        self.assertIn("delay", plan.topics)
        self.assertIn("risk", plan.topics)
        self.assertIn("Q4", plan.time_hints)
        self.assertIn("slipped", plan.expanded_terms)

    def test_chinese_coursework_request_gets_report_and_topic_metadata(self):
        plan = infer_intent("找出过去一年做过的计算机组成原理课设报告")
        self.assertEqual("locate_artifact", plan.intent)
        self.assertIn(".doc", plan.artifact_types)
        self.assertIn("computer_organization", plan.topics)
        self.assertIn("coursework", plan.topics)
        self.assertIn("past_year", plan.time_hints)

    def test_file_locator_groups_and_explains_semantic_matches(self):
        matches = self.retriever.locate_files("Find the presentation about customer delay and Q4 delivery risk")
        self.assertTrue(matches)
        self.assertTrue(matches[0].evidence)
        self.assertTrue(matches[0].reasons)
        self.assertTrue(any("semantic" in reason for reason in matches[0].reasons))

    def test_demo_semantic_request_finds_target_presentation(self):
        matches = self.retriever.locate_files("I wrote a PPT about customer delay and Q4 delivery risk, help me find it")
        self.assertEqual("Customer_Delivery_Risk_Q4_Final.pptx", matches[0].source)
        self.assertTrue(any("slide-level" in reason for reason in matches[0].reasons))

    def test_pptx_index_preserves_slide_locator(self):
        from pptx import Presentation

        with TemporaryDirectory() as directory:
            path = Path(directory) / "renewal-deck.pptx"
            presentation = Presentation()
            slide = presentation.slides.add_slide(presentation.slide_layouts[1])
            slide.shapes.title.text = "Renewal Proposal"
            slide.placeholders[1].text = "Customer pricing and Q4 decision timeline"
            presentation.save(path)
            retriever = LocalRetriever()
            retriever.index_paths([path])
        self.assertTrue(retriever.evidence[0].locator.startswith("slide 1"))
        self.assertIn("Renewal Proposal", retriever.evidence[0].text)

    def test_persistent_index_accumulates_and_records_changed_sources(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "decision.md"
            index_file = root / "workspace_index.json"
            source.write_text("first version", encoding="utf-8")
            first = LocalRetriever(index_file)
            first.index_paths([source], accumulate=True)
            first_hash = first.evidence[0].source_sha256
            reopened = LocalRetriever(index_file)
            self.assertEqual(1, len(reopened.evidence))
            source.write_text("changed version", encoding="utf-8")
            reopened.index_paths([source], accumulate=True)
            self.assertNotEqual(first_hash, reopened.evidence[0].source_sha256)
            self.assertTrue(any(item["status"] == "changed" for item in reopened.history))
            self.assertEqual("first version", reopened.archived_versions()[0].text)

    def test_local_embedding_and_query_history_are_optional_but_persistent(self):
        class FakeEmbedder:
            model = "fake-local-embed"

            def embed(self, texts):
                return [[float(len(text)), float(text.lower().count("q4")), float(text.lower().count("delay"))] for text in texts]

        with TemporaryDirectory() as directory:
            root = Path(directory)
            index_file = root / "workspace_index.json"
            retriever = LocalRetriever(index_file, embedder=FakeEmbedder())
            retriever.index_paths((Path(__file__).parent.parent / "demo_corpus").glob("*"))
            matches = retriever.locate_files("find my Q4 delay presentation")
            self.assertIsNotNone(retriever.embedding_matrix)
            self.assertEqual("fake-local-embed", retriever.query_history[-1]["embedding_model"])
            self.assertEqual("locate_artifact", retriever.query_history[-1]["intent"]["intent"])
            reopened = LocalRetriever(index_file)
            self.assertEqual(1, len(reopened.query_history))
            self.assertEqual(matches[0].source, reopened.query_history[0]["results"][0]["source"])

    def test_embedding_client_keeps_inputs_under_model_budget(self):
        client = LocalEmbeddingClient("vLLM ROCm", "http://127.0.0.1:8001/v1", "fake", max_chars=480)
        prepared = client._embedding_text("中" * 900)
        self.assertLessEqual(len(prepared), 480)

    def test_bm25_prioritizes_exact_chinese_phrase(self):
        index = BM25Index(["计算机组成原理课程设计报告", "数据库课程设计报告", "无关的会议记录"])
        scores = index.scores("计算机组成原理课设")
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[0], scores[2])

    def test_failed_local_embedding_endpoint_falls_back_to_hybrid_retrieval(self):
        class FailingEmbedder:
            model = "unavailable-local-embed"

            def embed(self, texts):
                raise ValueError("embedding service unavailable")

        retriever = LocalRetriever(embedder=FailingEmbedder())
        retriever.index_paths((Path(__file__).parent.parent / "demo_corpus").glob("*"))
        matches = retriever.locate_files("find the Q4 delivery presentation")
        self.assertTrue(matches)
        self.assertIsNone(retriever.embedding_matrix)
        self.assertIsNone(retriever.query_history[-1]["embedding_model"])

    def test_fts5_index_is_durable_and_parent_context_is_merged(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "long-record.md"
            source.write_text("intro " * 110 + "TARGET APPROVAL DECISION " + "follow-up " * 110, encoding="utf-8")
            index_file = root / "workspace_index.json"
            retriever = LocalRetriever(index_file)
            retriever.index_paths([source])
            self.assertTrue(retriever.fts5_active)
            self.assertTrue(retriever.evidence[0].parent_id)
            self.assertGreater(retriever.evidence[0].chunk_count, 1)
            reopened = LocalRetriever(index_file)
            results = reopened.search("TARGET APPROVAL DECISION")
            self.assertTrue(results)
            self.assertTrue(reopened.fts5_active)
            self.assertTrue(any(item.context_text for item in results))
            self.assertTrue(any("parent context merged" in reason for reason in results[0].match_reasons))

    def test_local_cross_encoder_reranker_can_promote_a_candidate(self):
        class FakeReranker:
            model = "fake-cross-encoder"

            def score(self, query, texts):
                return [1.0 if "preferred" in text else 0.0 for text in texts]

        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "ordinary.md"
            second = root / "preferred.md"
            first.write_text("common workspace record", encoding="utf-8")
            second.write_text("common workspace record preferred answer", encoding="utf-8")
            retriever = LocalRetriever(reranker=FakeReranker())
            retriever.index_paths([first, second])
            result = retriever.search("common workspace record", limit=1)
            self.assertEqual("preferred.md", result[0].source)
            self.assertEqual("fake-cross-encoder", retriever.query_history[-1]["reranker_model"])

    def test_scanned_pdf_page_uses_local_ocr_hook(self):
        from pypdf import PdfWriter

        with TemporaryDirectory() as directory:
            path = Path(directory) / "scan.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=300, height=300)
            with path.open("wb") as stream:
                writer.write(stream)
            with patch("core._ocr_pdf_page", return_value="OCR extracted local text"):
                sections = _document_sections(path)
        self.assertEqual("page 1 (OCR)", sections[0][0])
        self.assertIn("OCR extracted", sections[0][1])

    def test_championship_mode_rejects_missing_live_judge(self):
        evidence = self.retriever.search("99.9% uptime")
        with self.assertRaises(RuntimeError):
            run_court(
                "Did the vendor contractually commit to 99.9% uptime?",
                evidence,
                llm=None,
                allow_fallback=False,
            )


if __name__ == "__main__":
    unittest.main()
