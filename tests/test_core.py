from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from core import LocalRetriever, discover_workspace, infer_intent, markdown_brief, route_workspace_request, run_court, scan_sensitive_paths


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
        self.assertEqual(5, len(self.retriever.evidence))

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

    def test_file_locator_groups_and_explains_semantic_matches(self):
        matches = self.retriever.locate_files("Find the presentation about customer delay and Q4 delivery risk")
        self.assertTrue(matches)
        self.assertTrue(matches[0].evidence)
        self.assertTrue(matches[0].reasons)
        self.assertTrue(any("semantic" in reason for reason in matches[0].reasons))

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


if __name__ == "__main__":
    unittest.main()
