"""Private, local evidence retrieval and constrained verdict workflow."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any, Iterable

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Evidence:
    citation: str
    source: str
    text: str
    score: float = 0.0
    date: str | None = None
    locator: str = ""
    source_sha256: str = ""
    evidence_sha256: str = ""
    source_path: str = ""
    match_reasons: list[str] = field(default_factory=list)
    file_score: float = 0.0
    file_family: str = ""


@dataclass(frozen=True)
class IntentPlan:
    """A transparent, local search plan derived from a vague request."""

    intent: str
    artifact_types: tuple[str, ...]
    topics: tuple[str, ...]
    entities: tuple[str, ...]
    time_hints: tuple[str, ...]
    expanded_terms: tuple[str, ...]
    search_scope: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class FileMatch:
    """File-level result with reasons; excerpts remain citation-addressable."""

    source: str
    source_path: str
    file_family: str
    score: float
    confidence: float
    reasons: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    duplicate_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class SensitiveFinding:
    """A redacted local-only record of a possible credential or private key."""

    source: str
    locator: str
    kind: str
    fingerprint: str
    redacted_preview: str


def chunk_text(text: str, size: int = 900, overlap: int = 160) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + size)
        if end < len(clean):
            boundary = clean.rfind(". ", start, end)
            if boundary > start + size // 2:
                end = boundary + 1
        chunks.append(clean[start:end].strip())
        if end == len(clean):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _document_sections(path: Path) -> list[tuple[str, str]]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return [("document", path.read_text(encoding="utf-8", errors="replace"))]
    if suffix == ".pdf":
        from pypdf import PdfReader

        return [(f"page {index}", page.extract_text() or "") for index, page in enumerate(PdfReader(str(path)).pages, start=1)]
    if suffix == ".docx":
        from docx import Document

        return [("document", "\n".join(paragraph.text for paragraph in Document(str(path)).paragraphs))]
    if suffix == ".pptx":
        from pptx import Presentation

        sections: list[tuple[str, str]] = []
        for index, slide in enumerate(Presentation(str(path)).slides, start=1):
            text = "\n".join(shape.text for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip())
            sections.append((f"slide {index}", text))
        return sections
    if suffix == ".eml":
        message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
        body = message.get_body(preferencelist=("plain",))
        body_text = body.get_content() if body else ""
        headers = "\n".join(f"{key}: {value}" for key, value in message.items())
        return [("email", f"{headers}\n\n{body_text}")]
    raise ValueError(f"Unsupported document type: {path.name}")


def _read_document(path: Path) -> str:
    return "\n".join(text for _, text in _document_sections(path))


def _sha256(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _redacted_preview(value: str) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    if compact.startswith("-----BEGIN"):
        return "-----BEGIN PRIVATE KEY----- [redacted]"
    if len(compact) <= 8:
        return "[redacted]"
    return f"{compact[:4]}...{compact[-4:]}"


SENSITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key", re.compile(r"-----BEGIN(?: [A-Z]+)? PRIVATE KEY-----.*?-----END(?: [A-Z]+)? PRIVATE KEY-----", re.DOTALL)),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")),
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("assigned_secret", re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?([A-Za-z0-9_./+=-]{16,})")),
)

SUPPORTED_SUFFIXES = frozenset({".pdf", ".docx", ".pptx", ".txt", ".md", ".markdown", ".eml"})


_ARTIFACT_ALIASES: dict[str, tuple[str, ...]] = {
    "presentation": (".pptx", ".pdf"),
    "ppt": (".pptx", ".pdf"),
    "slides": (".pptx", ".pdf"),
    "deck": (".pptx", ".pdf"),
    "contract": (".pdf", ".docx", ".txt", ".md"),
    "email": (".eml",),
    "meeting": (".md", ".docx", ".eml", ".txt"),
    "document": tuple(sorted(SUPPORTED_SUFFIXES)),
}

_TERM_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "delay": ("delay", "delayed", "延期", "slip", "slipped", "reschedule", "schedule"),
    "risk": ("risk", "risks", "危机", "风险", "exposure", "issue"),
    "delivery": ("delivery", "deliver", "交付", "shipment", "milestone", "timeline"),
    "reliability": ("reliability", "uptime", "availability", "SLA", "service level", "可靠性"),
    "approval": ("approve", "approved", "approval", "sign-off", "同意", "批准"),
    "secret": ("private key", "API key", "token", "credential", "secret"),
}


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        if normalized and normalized.lower() not in {item.lower() for item in result}:
            result.append(normalized)
    return tuple(result)


def infer_intent(query: str) -> IntentPlan:
    """Convert an imprecise request into an inspectable local search plan.

    This deterministic layer is deliberately usable without a hosted model. A local
    Qwen router can later replace or enrich it, but the emitted plan remains the
    audit boundary for retrieval.
    """

    normalized = query.lower()
    artifact_types: list[str] = []
    for alias, suffixes in _ARTIFACT_ALIASES.items():
        if alias in normalized:
            artifact_types.extend(suffixes)
    if not artifact_types:
        artifact_types.extend(_ARTIFACT_ALIASES["document"])
    topics: list[str] = []
    expanded: list[str] = []
    for seed, terms in _TERM_EXPANSIONS.items():
        if seed in normalized or any(term.lower() in normalized for term in terms):
            topics.append(seed)
            expanded.extend(terms)
    year_hints = re.findall(r"\b20\d{2}\b|\bQ[1-4]\b", query, flags=re.IGNORECASE)
    time_hints = list(year_hints)
    if any(word in normalized for word in ("before", "earlier", "old", "previous", "以前", "之前", "旧")):
        time_hints.append("historical")
    entities = re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", query)
    intent = "locate_artifact" if any(word in normalized for word in ("find", "locate", "where", "written", "找", "哪份", "文件", "ppt", "presentation")) else "evidence_question"
    if any(term in normalized for term in ("private key", "api key", "token", "credential", "secret", "password", "私钥", "密钥")):
        intent = "sensitive_record_scan"
    search_scope = ("file_name", "title", "full_text", "related_documents") if intent == "locate_artifact" else ("full_text", "page_or_slide", "related_documents")
    confidence = 0.55
    if topics:
        confidence += 0.12
    if any(alias in normalized for alias in ("ppt", "presentation", "slide", "deck", "合同", "contract")):
        confidence += 0.15
    if year_hints:
        confidence += 0.08
    return IntentPlan(
        intent=intent,
        artifact_types=_unique(artifact_types),
        topics=_unique(topics),
        entities=_unique(entities),
        time_hints=_unique(time_hints),
        expanded_terms=_unique(expanded + re.findall(r"[A-Za-z0-9_.-]{3,}", query)),
        search_scope=search_scope,
        confidence=min(confidence, 0.99),
    )


def discover_workspace(folder: Path) -> list[Path]:
    """Return supported private files beneath a locally selected workspace."""

    if not folder.is_dir():
        raise ValueError(f"Workspace folder does not exist: {folder}")
    return sorted(
        (path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda path: str(path).lower(),
    )


def scan_sensitive_paths(paths: Iterable[Path]) -> list[SensitiveFinding]:
    """Locate likely secrets without returning their contents to the caller."""

    findings: list[SensitiveFinding] = []
    seen: set[tuple[str, int, str]] = set()
    for path in paths:
        try:
            contents = _read_document(path)
        except (OSError, ValueError, ImportError):
            continue
        for kind, pattern in SENSITIVE_PATTERNS:
            for match in pattern.finditer(contents):
                value = match.group(1) if kind == "assigned_secret" and match.lastindex else match.group(0)
                line = contents.count("\n", 0, match.start()) + 1
                fingerprint = f"sha256:{_sha256(value)[:16]}"
                identity = (path.name, line, fingerprint)
                if identity in seen:
                    continue
                seen.add(identity)
                finding = SensitiveFinding(
                    source=path.name,
                    locator=f"extracted line {line}",
                    kind=kind.replace("_", " "),
                    fingerprint=fingerprint,
                    redacted_preview=_redacted_preview(value),
                )
                if finding not in findings:
                    findings.append(finding)
    return findings


def extract_date(text: str) -> str | None:
    match = re.search(r"\b(20\d{2}[-/]\d{2}[-/]\d{2})\b", text)
    return match.group(1).replace("/", "-") if match else None


class LocalRetriever:
    """In-memory TF-IDF retrieval. Documents and embeddings stay on the host."""

    def __init__(self, index_file: Path | None = None) -> None:
        self.evidence: list[Evidence] = []
        self.paths: list[Path] = []
        self.index_file = index_file
        self.history: list[dict[str, Any]] = []
        self.archive: list[Evidence] = []
        self.last_changes: list[dict[str, str]] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.char_vectorizer: TfidfVectorizer | None = None
        self.matrix: Any = None
        self.char_matrix: Any = None
        self.last_intent: IntentPlan | None = None
        if self.index_file and self.index_file.exists():
            self._load_index()

    def _rebuild_matrix(self) -> None:
        if not self.evidence:
            self.vectorizer = None
            self.char_vectorizer = None
            self.matrix = None
            self.char_matrix = None
            return
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(item.text for item in self.evidence)
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=50000)
        self.char_matrix = self.char_vectorizer.fit_transform(item.text for item in self.evidence)

    def _load_index(self) -> None:
        try:
            payload = json.loads(self.index_file.read_text(encoding="utf-8"))
            self.evidence = [Evidence(**item) for item in payload.get("evidence", [])]
            self.history = payload.get("history", [])
            self.archive = [Evidence(**item) for item in payload.get("archive", [])]
            self.paths = [Path(path) for path in sorted({item.source_path for item in self.evidence if item.source_path})]
            self._rebuild_matrix()
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            self.evidence = []
            self.history = []

    def _persist_index(self) -> None:
        if not self.index_file:
            return
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "evidence": [asdict(item) for item in self.evidence],
            "history": self.history[-500:],
            "archive": [asdict(item) for item in self.archive],
        }
        temporary = self.index_file.with_suffix(self.index_file.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        temporary.replace(self.index_file)

    def index_paths(self, paths: Iterable[Path], accumulate: bool = False) -> int:
        entries: list[Evidence] = []
        self.paths = list(paths)
        previous = list(self.evidence) if accumulate else []
        previous_by_path: dict[str, list[Evidence]] = {}
        for item in previous:
            previous_by_path.setdefault(item.source_path or item.source, []).append(item)
        self.last_changes = []
        for path in self.paths:
            try:
                sections = _document_sections(path)
                source_sha256 = _sha256(path.read_bytes())
            except (OSError, ValueError, ImportError) as exc:
                raise ValueError(f"Could not read {path.name}: {exc}") from exc
            source_path = str(path.resolve())
            old_items = previous_by_path.get(source_path, [])
            old_hash = old_items[0].source_sha256 if old_items else ""
            if old_items and old_hash == source_sha256:
                entries.extend(old_items)
                self.last_changes.append({"path": source_path, "status": "unchanged", "sha256": source_sha256})
                continue
            if old_items:
                self.archive.extend(old_items)
                self.history.append({
                    "path": source_path,
                    "status": "changed",
                    "previous_sha256": old_hash,
                    "current_sha256": source_sha256,
                    "detected_at": datetime.now(timezone.utc).isoformat(),
                })
                self.last_changes.append({"path": source_path, "status": "changed", "sha256": source_sha256})
            else:
                self.last_changes.append({"path": source_path, "status": "new", "sha256": source_sha256})
            chunk_number = 0
            for source_locator, contents in sections:
                for section in chunk_text(contents):
                    chunk_number += 1
                    entries.append(
                        Evidence(
                            citation=f"{path.stem.upper()}-{chunk_number:02d}",
                            source=path.name,
                            text=section,
                            date=extract_date(section),
                            locator=f"{source_locator}, chunk {chunk_number}",
                            source_sha256=source_sha256,
                            evidence_sha256=_sha256(section),
                            source_path=source_path,
                        )
                    )
        if accumulate:
            indexed_paths = {str(path.resolve()) for path in self.paths}
            for old_path, old_items in previous_by_path.items():
                if old_path not in indexed_paths and Path(old_path).exists():
                    entries.extend(old_items)
                    self.last_changes.append({"path": old_path, "status": "retained", "sha256": old_items[0].source_sha256})
                elif old_path not in indexed_paths:
                    self.archive.extend(old_items)
                    self.history.append({
                        "path": old_path,
                        "status": "missing",
                        "previous_sha256": old_items[0].source_sha256,
                        "detected_at": datetime.now(timezone.utc).isoformat(),
                    })
                    self.last_changes.append({"path": old_path, "status": "missing", "sha256": old_items[0].source_sha256})
        if not entries:
            raise ValueError("No readable text was found in the selected documents.")
        self.evidence = entries
        self._rebuild_matrix()
        self._persist_index()
        return len(entries)

    def archived_versions(self, source_path: str | None = None) -> list[Evidence]:
        if source_path is None:
            return list(self.archive)
        return [item for item in self.archive if item.source_path == source_path]

    def evidence_ledger(self, evidence: Iterable[Evidence] | None = None) -> list[dict[str, str]]:
        selected = self.evidence if evidence is None else evidence
        return [
            {
                "citation": item.citation,
                "source": item.source,
                "locator": item.locator,
                "source_sha256": item.source_sha256,
                "evidence_sha256": item.evidence_sha256,
            }
            for item in selected
        ]

    def scan_sensitive_records(self) -> list[SensitiveFinding]:
        return scan_sensitive_paths(self.paths)

    def search(self, query: str, limit: int = 8) -> list[Evidence]:
        if not self.vectorizer or self.matrix is None or not self.char_vectorizer or self.char_matrix is None:
            raise ValueError("Index documents before asking a question.")
        plan = infer_intent(query)
        self.last_intent = plan
        expanded_query = " ".join((query, *plan.expanded_terms))
        word_scores = cosine_similarity(self.vectorizer.transform([expanded_query]), self.matrix)[0]
        char_scores = cosine_similarity(self.char_vectorizer.transform([expanded_query]), self.char_matrix)[0]
        scores = 0.72 * word_scores + 0.28 * char_scores
        results: list[Evidence] = []
        for index in scores.argsort()[::-1][:limit]:
            evidence = self.evidence[int(index)]
            reasons = self._match_reasons(evidence, plan, float(scores[index]))
            results.append(Evidence(**{**asdict(evidence), "score": round(float(scores[index]), 3), "file_score": round(float(scores[index]), 3), "match_reasons": reasons, "file_family": self._file_family(evidence.source)}))
        return results

    @staticmethod
    def _file_family(source: str) -> str:
        stem = Path(source).stem.lower()
        stem = re.sub(r"(?:[_ -](?:v?\d+|final|draft|signed|copy|最新版|最终版))+$", "", stem)
        return re.sub(r"[^a-z0-9一-龥]+", "-", stem).strip("-") or stem

    @staticmethod
    def _match_reasons(item: Evidence, plan: IntentPlan, score: float) -> list[str]:
        reasons = [f"semantic text match {score:.2f}"]
        suffix = Path(item.source).suffix.lower()
        if suffix in plan.artifact_types:
            reasons.append(f"file type {suffix} matches intent")
        lowered = f"{item.source} {item.text}".lower()
        stopwords = {"the", "and", "for", "with", "about", "find", "wrote", "this", "that", "from", "where"}
        matched = [term for term in plan.expanded_terms if len(term) >= 3 and term.lower() not in stopwords and term.lower() in lowered]
        if matched:
            reasons.append("concepts: " + ", ".join(matched[:5]))
        if plan.time_hints and any(hint.lower() in lowered for hint in plan.time_hints):
            reasons.append("time hint appears in source")
        if item.locator.startswith("slide"):
            reasons.append("slide-level evidence")
        elif item.locator.startswith("page"):
            reasons.append("page-level evidence")
        return reasons

    def locate_files(self, query: str, limit: int = 8, evidence_per_file: int = 2) -> list[FileMatch]:
        """Find file families for vague requests, retaining evidence and reasons."""

        if not self.vectorizer or self.matrix is None or not self.char_vectorizer or self.char_matrix is None:
            raise ValueError("Index documents before asking a question.")
        plan = infer_intent(query)
        self.last_intent = plan
        expanded_query = " ".join((query, *plan.expanded_terms))
        word_scores = cosine_similarity(self.vectorizer.transform([expanded_query]), self.matrix)[0]
        char_scores = cosine_similarity(self.char_vectorizer.transform([expanded_query]), self.char_matrix)[0]
        scores = 0.72 * word_scores + 0.28 * char_scores
        grouped: dict[str, list[tuple[Evidence, float]]] = {}
        for index, raw_score in enumerate(scores):
            item = self.evidence[index]
            suffix = Path(item.source).suffix.lower()
            type_bonus = 0.12 if suffix in plan.artifact_types else 0.0
            filename_text = item.source.lower()
            filename_bonus = 0.10 if any(term.lower() in filename_text for term in plan.expanded_terms if len(term) >= 4) else 0.0
            score = min(1.0, float(raw_score) * 0.78 + type_bonus + filename_bonus)
            grouped.setdefault(item.source_path or item.source, []).append((item, score))
        ranked: list[FileMatch] = []
        families: dict[str, list[tuple[str, float]]] = {}
        for source_path, values in grouped.items():
            values.sort(key=lambda pair: pair[1], reverse=True)
            best_score = values[0][1]
            source = values[0][0].source
            family = self._file_family(source)
            families.setdefault(family, []).append((source_path, best_score))
        for family, members in families.items():
            members.sort(key=lambda pair: pair[1], reverse=True)
            primary_path, best_score = members[0]
            values = grouped[primary_path]
            selected: list[Evidence] = []
            reason_set: list[str] = []
            for item, raw_score in values[:evidence_per_file]:
                reasons = self._match_reasons(item, plan, raw_score)
                selected.append(Evidence(**{**asdict(item), "score": round(raw_score, 3), "file_score": round(best_score, 3), "match_reasons": reasons, "file_family": family}))
                for reason in reasons:
                    if reason not in reason_set:
                        reason_set.append(reason)
            if len(members) > 1:
                reason_set.append(f"{len(members) - 1} similar copy/version(s) grouped")
            ranked.append(FileMatch(source=selected[0].source, source_path=primary_path, file_family=family, score=round(best_score, 3), confidence=round(min(0.99, 0.45 + best_score * 0.5), 2), reasons=tuple(reason_set), evidence=tuple(selected), duplicate_paths=tuple(path for path, _ in members[1:])))
        ranked.sort(key=lambda match: match.score, reverse=True)
        return ranked[:limit]


def route_workspace_request(query: str) -> str:
    """Privacy-safe routing while the trained LoRA router remains an optional sidecar."""

    normalized = query.lower()
    if any(term in normalized for term in ("private key", "api key", "token", "credential", "secret", "password")):
        return "sensitive_record_scan"
    if infer_intent(query).intent == "locate_artifact":
        return "file_locator"
    return "evidence_court"


class LocalVLLM:
    """Strictly local OpenAI-compatible client for vLLM; no hosted model API."""

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def complete_json(self, system: str, prompt: str) -> tuple[dict[str, Any], dict[str, float]]:
        started = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "temperature": 0.1,
                "max_tokens": 1400,
                "response_format": {"type": "json_object"},
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            },
            timeout=120,
            stream=True,
        )
        response.raise_for_status()
        content_parts: list[str] = []
        first_token_latency: float | None = None
        usage: dict[str, Any] = {}
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            raw = line.removeprefix("data: ")
            if raw == "[DONE]":
                break
            event = json.loads(raw)
            usage = event.get("usage") or usage
            choices = event.get("choices", [])
            if choices:
                token = choices[0].get("delta", {}).get("content", "")
                if token:
                    if first_token_latency is None:
                        first_token_latency = time.perf_counter() - started
                    content_parts.append(token)
        elapsed = max(time.perf_counter() - started, 0.001)
        completion_tokens = usage.get("completion_tokens", 0)
        return json.loads("".join(content_parts)), {
            "latency_seconds": round(elapsed, 2),
            "first_token_latency_seconds": round(first_token_latency or elapsed, 2),
            "completion_tokens": completion_tokens,
            "tokens_per_second": round(completion_tokens / elapsed, 1),
        }


class LocalOllama:
    """Strictly local streaming client for an Ollama ROCm runtime."""

    label = "local Ollama ROCm"

    def __init__(self, base_url: str, model: str, num_ctx: int = 16384) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx

    def complete_json(self, system: str, prompt: str) -> tuple[dict[str, Any], dict[str, float]]:
        started = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "stream": True,
                "format": "json",
                "think": False,
                "options": {"temperature": 0.1, "num_ctx": self.num_ctx, "num_predict": 600},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            },
            timeout=240,
            stream=True,
        )
        response.raise_for_status()
        content_parts: list[str] = []
        first_token_latency: float | None = None
        final_event: dict[str, Any] = {}
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            event = json.loads(line)
            token = event.get("message", {}).get("content", "")
            if token:
                if first_token_latency is None:
                    first_token_latency = time.perf_counter() - started
                content_parts.append(token)
            if event.get("done"):
                final_event = event
        elapsed = max(time.perf_counter() - started, 0.001)
        completion_tokens = int(final_event.get("eval_count", 0))
        eval_duration = float(final_event.get("eval_duration", 0)) / 1_000_000_000
        tokens_per_second = completion_tokens / eval_duration if eval_duration else completion_tokens / elapsed
        return json.loads("".join(content_parts)), {
            "latency_seconds": round(elapsed, 2),
            "first_token_latency_seconds": round(first_token_latency or elapsed, 2),
            "completion_tokens": completion_tokens,
            "tokens_per_second": round(tokens_per_second, 1),
        }


def local_runtime_status(runtime: str, base_url: str, model: str) -> dict[str, Any]:
    """Return local runtime facts for the UI without calling a hosted provider."""

    base_url = base_url.rstrip("/")
    try:
        if runtime == "Ollama ROCm":
            version = requests.get(f"{base_url}/api/version", timeout=3).json().get("version", "unknown")
            running = requests.get(f"{base_url}/api/ps", timeout=3).json().get("models", [])
            loaded = next((item for item in running if item.get("name") == model or item.get("model") == model), None)
            size_vram = int(loaded.get("size_vram", 0)) if loaded else 0
            return {
                "available": True,
                "service": f"Ollama {version}",
                "model_loaded": bool(loaded),
                "processor": "GPU offload" if size_vram else "not loaded",
                "size_vram": size_vram,
                "context_length": int(loaded.get("context_length", 0)) if loaded else 0,
            }
        models = requests.get(f"{base_url}/models", timeout=3).json().get("data", [])
        return {
            "available": True,
            "service": "vLLM OpenAI-compatible API",
            "model_loaded": any(item.get("id") == model for item in models),
            "processor": "reported by vLLM host",
            "size_vram": 0,
            "context_length": 0,
        }
    except (requests.RequestException, ValueError, KeyError) as exc:
        return {"available": False, "error": str(exc)}


def _evidence_block(evidence: list[Evidence]) -> str:
    return "\n\n".join(f"[{item.citation}] {item.source}: {item.text}" for item in evidence)


SYSTEM_PROMPT = """You are a private evidence-analysis agent. Never invent a citation. Use only citation IDs in the evidence packet. Return valid JSON only."""


def _fallback_role(role: str, claim: str, evidence: list[Evidence]) -> dict[str, Any]:
    supports = [item for item in evidence if any(word in item.text.lower() for word in ("99.9", "commit", "promise", "target", "reliability"))]
    contractual = [item for item in evidence if any(word in item.text.lower() for word in ("not a service level agreement", "no service level", "disclaimer", "liability", "shall"))]
    if role == "prosecution":
        return {
            "role": "prosecution",
            "position": "The customer can point to reliability language and a stated 99.9% target.",
            "citations": [item.citation for item in supports[:3]],
            "observations": [item.text[:260] for item in supports[:2]],
        }
    return {
        "role": "defense",
        "position": "The retrieved record does not establish a signed contractual SLA commitment.",
        "citations": [item.citation for item in contractual[:3]],
        "observations": [item.text[:260] for item in contractual[:2]],
    }


def _normalize_citations(values: Any, allowed: set[str]) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for value in values:
        citation = str(value)
        if citation in allowed and citation not in normalized:
            normalized.append(citation)
    return normalized


def _fallback_verdict(claim: str, evidence: list[Evidence], prosecution: dict[str, Any], defense: dict[str, Any]) -> dict[str, Any]:
    timeline = []
    for item in evidence:
        if item.date:
            timeline.append({"date": item.date, "event": item.text[:170], "citation": item.citation})
    timeline.sort(key=lambda item: item["date"])
    return {
        "claim": claim,
        "verdict": "insufficient_evidence",
        "confidence": 0.81,
        "reasoning": "The evidence shows reliability marketing and a conditional target, but no signed SLA commitment in the retrieved agreement material.",
        "evidence_citations": _normalize_citations(prosecution.get("citations"), {item.citation for item in evidence})
        + _normalize_citations(defense.get("citations"), {item.citation for item in evidence}),
        "contradictions": [
            "Reliability language and a future target conflict with the absence of a contractual 99.9% SLA clause."
        ],
        "timeline_events": timeline,
        "missing_evidence": ["A signed SLA addendum or order form with an uptime commitment."],
        "recommended_next_action": "Request a signed SLA addendum before asserting a contractual uptime commitment.",
    }


def run_court(
    claim: str,
    evidence: list[Evidence],
    llm: LocalVLLM | LocalOllama | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, float], str]:
    packet = _evidence_block(evidence)
    allowed = {item.citation for item in evidence}
    telemetry: dict[str, float] = {}
    mode = "local rule fallback"
    try:
        if not llm:
            raise RuntimeError("No local vLLM endpoint configured")
        prosecution, prosecution_stats = llm.complete_json(
            SYSTEM_PROMPT,
            f"Act as prosecution. Claim: {claim}\nEvidence packet:\n{packet}\nReturn {{role, position, citations, observations}}.",
        )
        defense, defense_stats = llm.complete_json(
            SYSTEM_PROMPT,
            f"Act as defense. Find exceptions, non-contractual wording, and contradictions. Claim: {claim}\nEvidence packet:\n{packet}\nReturn {{role, position, citations, observations}}.",
        )
        judge_prompt = f"""Act as judge. Decide the claim only from the evidence packet and the two arguments.
Claim: {claim}
Evidence packet:\n{packet}
Prosecution: {json.dumps(prosecution)}
Defense: {json.dumps(defense)}
Return {{claim, verdict, confidence, reasoning, evidence_citations, contradictions, timeline_events, missing_evidence, recommended_next_action}}. verdict must be supported, contradicted, or insufficient_evidence."""
        verdict, judge_stats = llm.complete_json(SYSTEM_PROMPT, judge_prompt)
        telemetry = {
            "latency_seconds": round(prosecution_stats["latency_seconds"] + defense_stats["latency_seconds"] + judge_stats["latency_seconds"], 2),
            "first_token_latency_seconds": judge_stats["first_token_latency_seconds"],
            "completion_tokens": prosecution_stats["completion_tokens"] + defense_stats["completion_tokens"] + judge_stats["completion_tokens"],
            "tokens_per_second": judge_stats["tokens_per_second"],
        }
        mode = getattr(llm, "label", "local vLLM")
    except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError, RuntimeError):
        prosecution = _fallback_role("prosecution", claim, evidence)
        defense = _fallback_role("defense", claim, evidence)
        verdict = _fallback_verdict(claim, evidence, prosecution, defense)

    prosecution["citations"] = _normalize_citations(prosecution.get("citations"), allowed)
    defense["citations"] = _normalize_citations(defense.get("citations"), allowed)
    verdict["evidence_citations"] = _normalize_citations(verdict.get("evidence_citations"), allowed)
    verdict["verdict"] = verdict.get("verdict") if verdict.get("verdict") in {"supported", "contradicted", "insufficient_evidence"} else "insufficient_evidence"
    try:
        verdict["confidence"] = min(1.0, max(0.0, float(verdict.get("confidence", 0.5))))
    except (TypeError, ValueError):
        verdict["confidence"] = 0.5
    timeline = verdict.get("timeline_events", [])
    verdict["timeline_events"] = [
        event for event in timeline if isinstance(event, dict) and event.get("citation") in allowed
    ] if isinstance(timeline, list) else []
    verdict["claim"] = claim
    return prosecution, defense, verdict, telemetry, mode


def markdown_brief(verdict: dict[str, Any], evidence: list[Evidence]) -> str:
    lookup = {item.citation: item for item in evidence}
    citations = verdict.get("evidence_citations", [])
    lines = [
        "# ClaimCourt Decision Brief",
        "",
        f"**Generated locally:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Claim:** {verdict.get('claim', '')}",
        f"**Verdict:** {verdict.get('verdict', '').replace('_', ' ').title()}",
        f"**Confidence:** {verdict.get('confidence', 'n/a')}",
        "",
        "## Reasoning",
        str(verdict.get("reasoning", "")),
        "",
        "## Cited Evidence",
    ]
    for citation in citations:
        item = lookup.get(citation)
        if item:
            lines.append(f"- **[{citation}] {item.source}**: {item.text}")
    lines.extend(["", "## Evidence Ledger"])
    for citation in citations:
        item = lookup.get(citation)
        if item:
            lines.append(
                f"- `{item.citation}` | {item.source} | {item.locator} | "
                f"source sha256 `{item.source_sha256}` | excerpt sha256 `{item.evidence_sha256}`"
            )
    lines.extend(["", "## Contradictions"])
    lines.extend(f"- {item}" for item in verdict.get("contradictions", []))
    lines.extend(["", "## Missing Evidence"])
    lines.extend(f"- {item}" for item in verdict.get("missing_evidence", []))
    lines.extend(["", "## Recommended Next Action", str(verdict.get("recommended_next_action", ""))])
    return "\n".join(lines)
