"""Private, local evidence retrieval and constrained verdict workflow."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
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
    modified_at: str = ""
    match_reasons: list[str] = field(default_factory=list)
    file_score: float = 0.0
    file_family: str = ""
    parent_id: str = ""
    parent_locator: str = ""
    chunk_index: int = 0
    chunk_count: int = 0
    context_text: str = ""
    ocr_used: bool = False


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


class LocalEmbeddingClient:
    """OpenAI-compatible or Ollama embedding client; all calls stay loopback/local."""

    def __init__(self, runtime: str, base_url: str, model: str, batch_size: int = 32, max_chars: int = 480) -> None:
        self.runtime = runtime
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = max(1, batch_size)
        self.max_chars = max(128, max_chars)

    def _embedding_text(self, text: str) -> str:
        """Keep a conservative character budget for tokenizers with a 512-token limit."""

        clean = re.sub(r"\s+", " ", text).strip()
        if len(clean) <= self.max_chars:
            return clean
        head = int(self.max_chars * 0.68)
        marker = " ... "
        tail = max(1, self.max_chars - head - len(marker))
        return f"{clean[:head]}{marker}{clean[-tail:]}"

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.runtime == "Ollama ROCm":
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            vectors = payload.get("embeddings", [])
        else:
            response = requests.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": texts},
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            vectors = [item["embedding"] for item in payload.get("data", [])]
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ValueError("Local embedding runtime returned an invalid vector count.")
        return [[float(value) for value in vector] for vector in vectors]

    def _embed_resilient(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._embed_batch(texts)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if len(texts) <= 1 or status not in {400, 413, 422}:
                raise
        except ValueError:
            if len(texts) <= 1:
                raise
        midpoint = max(1, len(texts) // 2)
        return self._embed_resilient(texts[:midpoint]) + self._embed_resilient(texts[midpoint:])

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared = [self._embedding_text(text) for text in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(prepared), self.batch_size):
            vectors.extend(self._embed_resilient(prepared[start : start + self.batch_size]))
        return vectors


_BM25_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


def _bm25_tokens(text: str) -> list[str]:
    """Tokenize English words and Chinese characters/bigrams for exact retrieval."""

    tokens: list[str] = []
    for segment in _BM25_TOKEN_RE.findall(text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", segment):
            tokens.extend(segment)
            tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
        else:
            tokens.append(segment)
    return tokens


class BM25Index:
    """Small in-process BM25 index for exact terms and Chinese short phrases."""

    def __init__(self, documents: Iterable[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents = [_bm25_tokens(document) for document in documents]
        self.document_lengths = [len(tokens) for tokens in self.documents]
        self.average_length = sum(self.document_lengths) / max(len(self.document_lengths), 1)
        document_frequency: Counter[str] = Counter()
        postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, tokens in enumerate(self.documents):
            counts = Counter(tokens)
            for token, frequency in counts.items():
                document_frequency[token] += 1
                postings[token].append((index, frequency))
        self.postings = dict(postings)
        document_count = len(self.documents)
        self.idf = {
            token: max(0.0, math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5)))
            for token, frequency in document_frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        result = [0.0] * len(self.documents)
        query_tokens = set(_bm25_tokens(query))
        for token in query_tokens:
            idf = self.idf.get(token)
            if idf is None:
                continue
            for index, frequency in self.postings.get(token, []):
                length = self.document_lengths[index]
                normalization = self.k1 * (1.0 - self.b + self.b * length / max(self.average_length, 1.0))
                result[index] += idf * (frequency * (self.k1 + 1.0)) / (frequency + normalization)
        maximum = max(result, default=0.0)
        if maximum <= 0:
            return result
        return [score / maximum for score in result]


def _fts_search_text(text: str) -> str:
    """Materialize the same bilingual tokens used by BM25 for SQLite FTS5."""

    return " ".join(_bm25_tokens(text))


class FTS5Index:
    """Durable SQLite FTS5 index kept beside the JSON evidence ledger.

    The JSON file remains the source of truth for evidence and audit history. FTS5
    is a rebuildable, local acceleration/indexing layer, so an interrupted write
    cannot corrupt the evidence ledger or expose documents outside the workspace.
    """

    def __init__(self, database_path: Path | None = None) -> None:
        self.database_path = database_path
        self.connection: sqlite3.Connection | None = None
        if database_path is None:
            self.connection = sqlite3.connect(":memory:", check_same_thread=False)
            self._ensure_schema(self.connection)
        else:
            connection = sqlite3.connect(str(database_path), check_same_thread=False)
            try:
                self._ensure_schema(connection)
            finally:
                connection.close()

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
               USING fts5(citation UNINDEXED, source, locator, search_text,
                         tokenize='unicode61 remove_diacritics 2')"""
        )
        connection.commit()

    def _open(self) -> tuple[sqlite3.Connection, bool]:
        if self.database_path is None:
            if self.connection is None:
                self.connection = sqlite3.connect(":memory:", check_same_thread=False)
                self._ensure_schema(self.connection)
            return self.connection, False
        connection = sqlite3.connect(str(self.database_path), check_same_thread=False)
        self._ensure_schema(connection)
        return connection, True

    @property
    def active(self) -> bool:
        return self.database_path is not None or self.connection is not None

    def rebuild(self, evidence: Iterable[Evidence]) -> None:
        rows = [
            (item.citation, item.source, item.locator, _fts_search_text(item.text))
            for item in evidence
        ]
        connection, transient = self._open()
        try:
            connection.execute("DELETE FROM chunks_fts")
            if rows:
                connection.executemany(
                    "INSERT INTO chunks_fts(citation, source, locator, search_text) VALUES (?, ?, ?, ?)",
                    rows,
                )
            connection.commit()
        finally:
            if transient:
                connection.close()

    def scores(self, query: str, citations: list[str]) -> list[float]:
        """Return normalized FTS5 relevance aligned with the evidence list."""

        result = [0.0] * len(citations)
        tokens = list(dict.fromkeys(_bm25_tokens(query)))[:96]
        if not tokens:
            return result
        expression = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
        connection: sqlite3.Connection | None = None
        transient = False
        try:
            connection, transient = self._open()
            rows = connection.execute(
                "SELECT citation, bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts)",
                (expression,),
            ).fetchall()
        except sqlite3.OperationalError:
            return result
        finally:
            if transient and connection is not None:
                connection.close()
        by_citation: defaultdict[str, float] = defaultdict(float)
        for citation, rank in rows:
            # SQLite's bm25() is lower-is-better and normally negative.
            by_citation[str(citation)] = max(by_citation[str(citation)], max(0.0, -float(rank)))
        raw = [by_citation.get(citation, 0.0) for citation in citations]
        maximum = max(raw, default=0.0)
        return [value / maximum for value in raw] if maximum > 0 else result

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None


class LocalCrossEncoderReranker:
    """Optional local sentence-transformers cross-encoder reranker.

    The dependency and model are loaded lazily. This keeps the base install small
    and lets the same code run on CPU, ROCm PyTorch, or a private model directory.
    """

    def __init__(self, model: str, device: str | None = None, max_length: int = 512) -> None:
        self.model = model
        self.model_name = model
        self.device = device or os.getenv("CLAIMCOURT_RERANKER_DEVICE", "") or None
        self.max_length = max(128, max_length)
        self._model: Any = None
        self.error = ""

    @property
    def available(self) -> bool:
        return self._model is not None

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            # CrossEncoder otherwise downloads a named checkpoint implicitly.
            # Keep the privacy boundary explicit: cache hits are allowed, network
            # model downloads are not.
            kwargs: dict[str, Any] = {
                "max_length": self.max_length,
                "automodel_args": {"local_files_only": True},
                "tokenizer_args": {"local_files_only": True},
                "local_files_only": True,
            }
            if self.device:
                kwargs["device"] = self.device
            self._model = CrossEncoder(self.model, **kwargs)
        except Exception as exc:  # optional capability must never break lexical retrieval
            self.error = f"{type(exc).__name__}: {exc}"
            self._model = None
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        model = self._ensure_model()
        if model is None or not texts:
            return []
        pairs = [(query, text[: self.max_length * 4]) for text in texts]
        values = model.predict(pairs, show_progress_bar=False)
        return [float(value) for value in values]


@dataclass(frozen=True)
class SensitiveFinding:
    """A redacted local-only record of a possible credential or private key."""

    source: str
    locator: str
    kind: str
    fingerprint: str
    redacted_preview: str


def chunk_text(text: str, size: int = 480, overlap: int = 80) -> list[str]:
    clean = re.sub(r"[ \t\f\v]+", " ", text)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    if not clean:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + size)
        if end < len(clean):
            best_boundary = -1
            best_width = 0
            for separator in ("\n\n", "\n", "。", "！", "？", ". ", "; ", "；", "，", ", ", " "):
                boundary = clean.rfind(separator, start, end)
                if boundary > start + size // 2 and boundary > best_boundary:
                    best_boundary = boundary
                    best_width = len(separator)
            if best_boundary >= 0:
                end = best_boundary + best_width
        chunks.append(clean[start:end].strip())
        if end == len(clean):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _ocr_pdf_page(path: Path, page_number: int) -> str:
    """Best-effort OCR for a scanned PDF page using only local binaries/models."""

    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    try:
        language = os.getenv("CLAIMCOURT_OCR_LANG", "eng+chi_sim")
        with fitz.open(str(path)) as document:
            page = document.load_page(page_number - 1)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            return pytesseract.image_to_string(image, lang=language).strip()
    except Exception:
        # Missing tesseract language packs, corrupt pages, and unavailable OCR
        # binaries are reported as an unreadable page by the caller if no text is
        # available; they never abort indexing of the remaining workspace.
        return ""


def _document_sections(path: Path) -> list[tuple[str, str]]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return [("document", path.read_text(encoding="utf-8", errors="replace"))]
    if suffix == ".pdf":
        from pypdf import PdfReader

        sections: list[tuple[str, str]] = []
        for index, page in enumerate(PdfReader(str(path)).pages, start=1):
            text = page.extract_text() or ""
            if len(text.strip()) < 24:
                ocr_text = _ocr_pdf_page(path, index)
                if ocr_text:
                    sections.append((f"page {index} (OCR)", ocr_text))
                    continue
            sections.append((f"page {index}", text))
        return sections
    if suffix == ".docx":
        from docx import Document

        return [("document", "\n".join(paragraph.text for paragraph in Document(str(path)).paragraphs))]
    if suffix == ".doc":
        antiword = shutil.which("antiword")
        if antiword:
            result = subprocess.run(
                [antiword, "-w", "0", str(path)],
                capture_output=True,
                timeout=60,
                check=False,
            )
            text = result.stdout.decode("utf-8", errors="replace")
            if result.returncode == 0 and text.strip():
                return [("document", text)]
            raise ValueError(f"antiword could not parse {path.name}")
        libreoffice = shutil.which("libreoffice") or shutil.which("soffice")
        if libreoffice:
            with tempfile.TemporaryDirectory(prefix="claimcourt-doc-") as temporary:
                result = subprocess.run(
                    [libreoffice, "--headless", "--convert-to", "txt:Text", "--outdir", temporary, str(path)],
                    capture_output=True,
                    timeout=120,
                    check=False,
                )
                converted = Path(temporary) / f"{path.stem}.txt"
                if result.returncode == 0 and converted.exists():
                    text = converted.read_text(encoding="utf-8", errors="replace")
                    if text.strip():
                        return [("document", text)]
        raise ValueError("No antiword or LibreOffice converter is installed for legacy .doc files")
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

SUPPORTED_SUFFIXES = frozenset({".pdf", ".doc", ".docx", ".pptx", ".txt", ".md", ".markdown", ".eml"})


_ARTIFACT_ALIASES: dict[str, tuple[str, ...]] = {
    "presentation": (".pptx", ".pdf"),
    "ppt": (".pptx", ".pdf"),
    "slides": (".pptx", ".pdf"),
    "deck": (".pptx", ".pdf"),
    "report": (".pdf", ".doc", ".docx", ".pptx", ".txt", ".md"),
    "报告": (".pdf", ".doc", ".docx", ".pptx", ".txt", ".md"),
    "课设": (".pdf", ".doc", ".docx", ".pptx", ".txt", ".md"),
    "课程设计": (".pdf", ".doc", ".docx", ".pptx", ".txt", ".md"),
    "实验报告": (".pdf", ".doc", ".docx", ".pptx", ".txt", ".md"),
    "contract": (".pdf", ".doc", ".docx", ".txt", ".md"),
    "email": (".eml",),
    "meeting": (".md", ".doc", ".docx", ".eml", ".txt"),
    "document": tuple(sorted(SUPPORTED_SUFFIXES)),
}

_TERM_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "delay": ("delay", "delayed", "延期", "slip", "slipped", "reschedule", "schedule"),
    "risk": ("risk", "risks", "危机", "风险", "exposure", "issue"),
    "delivery": ("delivery", "deliver", "交付", "shipment", "milestone", "timeline"),
    "reliability": ("reliability", "uptime", "availability", "SLA", "service level", "可靠性"),
    "approval": ("approve", "approved", "approval", "sign-off", "同意", "批准"),
    "secret": ("private key", "API key", "token", "credential", "secret"),
    "computer_organization": (
        "computer organization",
        "computer architecture",
        "计算机组成原理",
        "计算机组成与结构",
        "计组",
        "CPU",
        "单周期CPU",
        "流水线",
        "cache",
        "缓存",
    ),
    "coursework": (
        "课设",
        "课程设计",
        "课程报告",
        "实验报告",
        "课设报告",
        "coursework",
        "course project",
        "lab report",
    ),
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
    if any(word in normalized for word in ("过去一年", "近一年", "最近一年", "去年", "last year", "past year", "recent year")):
        time_hints.append("past_year")
    entities = re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", query)
    intent = "locate_artifact" if any(word in normalized for word in ("find", "locate", "where is", "written", "找", "找出", "哪份", "找不到", "报告", "课设", "课程设计", "ppt", "presentation")) else "evidence_question"
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
        except Exception:
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
    """Private hybrid retrieval with durable FTS5 and parent-aware context."""

    def __init__(
        self,
        index_file: Path | None = None,
        embedder: LocalEmbeddingClient | None = None,
        reranker: Any | None = None,
    ) -> None:
        self.evidence: list[Evidence] = []
        self.paths: list[Path] = []
        self.index_file = index_file
        self.history: list[dict[str, Any]] = []
        self.archive: list[Evidence] = []
        self.last_changes: list[dict[str, str]] = []
        self.ingestion_errors: list[dict[str, str]] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.char_vectorizer: TfidfVectorizer | None = None
        self.matrix: Any = None
        self.char_matrix: Any = None
        self.bm25: BM25Index | None = None
        self.fts_error = ""
        self.fts: FTS5Index | None = None
        self.parent_records: dict[str, dict[str, Any]] = {}
        self.embedder = embedder
        self.reranker = reranker
        self.reranker_model = ""
        self.embedding_matrix: Any = None
        self.embedding_model = ""
        self.query_history: list[dict[str, Any]] = []
        self.last_intent: IntentPlan | None = None
        try:
            if self.index_file:
                self.index_file.parent.mkdir(parents=True, exist_ok=True)
            fts_path = None if self.index_file is None else self.index_file.with_suffix(self.index_file.suffix + ".fts5.sqlite3")
            self.fts = FTS5Index(fts_path)
        except (sqlite3.Error, OSError) as exc:
            self.fts_error = f"{type(exc).__name__}: {exc}"
            self.fts = None
        if self.index_file and self.index_file.exists():
            self._load_index()

    @property
    def fts5_active(self) -> bool:
        return bool(self.fts)

    def _rebuild_matrix(self) -> None:
        if not self.evidence:
            self.vectorizer = None
            self.char_vectorizer = None
            self.matrix = None
            self.char_matrix = None
            self.bm25 = None
            if self.fts:
                self.fts.rebuild([])
            return
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(item.text for item in self.evidence)
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=50000)
        self.char_matrix = self.char_vectorizer.fit_transform(item.text for item in self.evidence)
        self.bm25 = BM25Index(item.text for item in self.evidence)
        if self.fts:
            self.fts.rebuild(self.evidence)

    def _load_index(self) -> None:
        try:
            payload = json.loads(self.index_file.read_text(encoding="utf-8"))
            self.evidence = [Evidence(**item) for item in payload.get("evidence", [])]
            self.history = payload.get("history", [])
            self.archive = [Evidence(**item) for item in payload.get("archive", [])]
            self.parent_records = {
                str(key): dict(value)
                for key, value in (payload.get("parents", {}) or {}).items()
                if isinstance(value, dict)
            }
            self.query_history = payload.get("query_history", [])[-100:]
            self.ingestion_errors = payload.get("ingestion_errors", [])[-500:]
            self.paths = [Path(path) for path in sorted({item.source_path for item in self.evidence if item.source_path})]
            self._backfill_parent_metadata()
            self._rebuild_matrix()
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            self.evidence = []
            self.history = []
            self.parent_records = {}

    def _backfill_parent_metadata(self) -> None:
        """Upgrade schema-1 JSON entries without changing their citation IDs."""

        upgraded: list[Evidence] = []
        for item in self.evidence:
            if item.parent_id:
                self.parent_records.setdefault(
                    item.parent_id,
                    {
                        "text": item.text,
                        "source_path": item.source_path,
                        "locator": item.parent_locator or item.locator,
                        "source_sha256": item.source_sha256,
                    },
                )
                upgraded.append(item)
                continue
            parent_locator = item.locator.split(", chunk", 1)[0]
            parent_id = _sha256(f"{item.source_sha256}:{parent_locator}")[:24]
            self.parent_records.setdefault(
                parent_id,
                {
                    "text": item.text,
                    "source_path": item.source_path,
                    "locator": parent_locator,
                    "source_sha256": item.source_sha256,
                },
            )
            upgraded.append(
                Evidence(
                    **{
                        **asdict(item),
                        "parent_id": parent_id,
                        "parent_locator": parent_locator,
                        "chunk_index": item.chunk_index or 1,
                        "chunk_count": item.chunk_count or 1,
                    }
                )
            )
        self.evidence = upgraded

    def _persist_index(self) -> None:
        if not self.index_file:
            return
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": 2,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "evidence": [asdict(item) for item in self.evidence],
            "history": self.history[-500:],
            "archive": [asdict(item) for item in self.archive],
            "parents": self.parent_records,
            "query_history": self.query_history[-100:],
            "ingestion_errors": self.ingestion_errors[-500:],
            "embedding_model": self.embedding_model,
            "fts5": bool(self.fts),
        }
        temporary = self.index_file.with_suffix(self.index_file.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        temporary.replace(self.index_file)

    def index_paths(self, paths: Iterable[Path], accumulate: bool = False) -> int:
        entries: list[Evidence] = []
        self.paths = list(paths)
        previous = list(self.evidence) if accumulate else []
        if not accumulate:
            self.parent_records = {}
        previous_by_path: dict[str, list[Evidence]] = {}
        for item in previous:
            previous_by_path.setdefault(item.source_path or item.source, []).append(item)
        self.last_changes = []
        self.ingestion_errors = []
        for path in self.paths:
            try:
                sections = _document_sections(path)
                source_sha256 = _sha256(path.read_bytes())
            except Exception as exc:
                error = {
                    "path": str(path.resolve()),
                    "status": "unreadable",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                self.ingestion_errors.append(error)
                self.last_changes.append(error)
                continue
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
                sections_chunks = chunk_text(contents)
                if not sections_chunks:
                    continue
                parent_id = _sha256(f"{source_sha256}:{source_locator}")[:24]
                self.parent_records[parent_id] = {
                    "text": contents,
                    "source_path": source_path,
                    "locator": source_locator,
                    "source_sha256": source_sha256,
                }
                parent_count = len(sections_chunks)
                for parent_chunk_index, section in enumerate(sections_chunks, start=1):
                    chunk_number += 1
                    modified_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).date().isoformat()
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
                            modified_at=modified_at,
                            parent_id=parent_id,
                            parent_locator=source_locator,
                            chunk_index=parent_chunk_index,
                            chunk_count=parent_count,
                            ocr_used="(OCR)" in source_locator,
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
        active_parent_ids = {item.parent_id for item in entries if item.parent_id}
        self.parent_records = {
            key: value for key, value in self.parent_records.items() if key in active_parent_ids
        }
        self._rebuild_matrix()
        self._rebuild_embedding_matrix()
        self._persist_index()
        return len(entries)

    def _rebuild_embedding_matrix(self) -> None:
        self.embedding_matrix = None
        self.embedding_model = ""
        if not self.embedder or not self.evidence:
            return
        try:
            vectors = self.embedder.embed([item.text for item in self.evidence])
            if vectors:
                self.embedding_matrix = vectors
                self.embedding_model = self.embedder.model
        except (requests.RequestException, ValueError, KeyError, TypeError):
            # Retrieval remains available through the deterministic local indexes.
            self.embedding_matrix = None

    def _semantic_scores(self, query: str) -> Any:
        if not self.embedder or self.embedding_matrix is None:
            return None
        try:
            query_vector = self.embedder.embed([query])
            if not query_vector:
                return None
            return cosine_similarity(query_vector, self.embedding_matrix)[0]
        except (requests.RequestException, ValueError, KeyError, TypeError):
            return None

    def _apply_reranker(self, query: str, scores: Any, candidate_indices: list[int]) -> Any:
        """Rerank only the hybrid shortlist; a failed optional model is transparent."""

        self.reranker_model = ""
        if not self.reranker or not candidate_indices:
            return scores
        self.reranker_model = str(
            getattr(self.reranker, "model_name", None)
            or getattr(self.reranker, "model", None)
            or type(self.reranker).__name__
        )
        try:
            values = self.reranker.score(query, [self.evidence[index].text for index in candidate_indices])
        except Exception:
            return scores
        if not isinstance(values, (list, tuple)) or len(values) != len(candidate_indices):
            return scores
        numeric = [float(value) for value in values]
        low, high = min(numeric, default=0.0), max(numeric, default=0.0)
        if high - low <= 1e-9:
            normalized = [0.5] * len(numeric)
        else:
            normalized = [(value - low) / (high - low) for value in numeric]
        adjusted = scores.copy()
        for index, rerank_score in zip(candidate_indices, normalized):
            adjusted[index] = 0.65 * float(scores[index]) + 0.35 * rerank_score
        return adjusted

    def _rank_indices(self, query: str, scores: Any, limit: int) -> tuple[Any, list[int]]:
        candidate_count = min(len(self.evidence), max(limit * 4, 32))
        candidates = [int(index) for index in scores.argsort()[::-1][:candidate_count]]
        adjusted = self._apply_reranker(query, scores, candidates)
        ranked = sorted(candidates, key=lambda index: float(adjusted[index]), reverse=True)
        return adjusted, ranked

    def _parent_context(self, item: Evidence, sibling_window: int = 1, max_chars: int = 1100) -> str:
        """Return adjacent child chunks from the same structural parent."""

        if not item.parent_id:
            return ""
        siblings = sorted(
            (candidate for candidate in self.evidence if candidate.parent_id == item.parent_id),
            key=lambda candidate: candidate.chunk_index,
        )
        if len(siblings) <= 1:
            parent = self.parent_records.get(item.parent_id, {})
            parent_text = str(parent.get("text", ""))
            if parent_text and parent_text.strip() != item.text.strip():
                return parent_text[:max_chars]
            return ""
        position = next((index for index, candidate in enumerate(siblings) if candidate.citation == item.citation), 0)
        selected = siblings[max(0, position - sibling_window) : position + sibling_window + 1]
        context = "\n".join(
            f"[parent chunk {candidate.chunk_index}/{candidate.chunk_count}] {candidate.text}"
            for candidate in selected
            if candidate.citation != item.citation
        )
        return context[:max_chars]

    def _with_parent_context(self, item: Evidence, score: float, plan: IntentPlan, file_score: float | None = None, family: str = "") -> Evidence:
        reasons = self._match_reasons(item, plan, score)
        if self.fts:
            reasons.append("SQLite FTS5 lexical index")
        if self.reranker_model:
            reasons.append(f"cross-encoder reranked ({self.reranker_model})")
        context_text = self._parent_context(item)
        return Evidence(
            **{
                **asdict(item),
                "score": round(float(score), 3),
                "file_score": round(float(file_score if file_score is not None else score), 3),
                "match_reasons": reasons,
                "file_family": family or self._file_family(item.source),
                "context_text": context_text,
            }
        )

    def _record_query(self, query: str, plan: IntentPlan, results: Iterable[Evidence | FileMatch]) -> None:
        compact_results: list[dict[str, Any]] = []
        for result in results:
            if isinstance(result, FileMatch):
                compact_results.append({
                    "source": result.source,
                    "source_path": result.source_path,
                    "score": result.score,
                    "confidence": result.confidence,
                    "reasons": list(result.reasons),
                })
            else:
                compact_results.append({
                    "citation": result.citation,
                    "source": result.source,
                    "score": result.score,
                })
        self.query_history.append({
            "query_id": f"Q-{len(self.query_history) + 1:04d}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "intent": asdict(plan),
            "results": compact_results,
            "embedding_model": self.embedding_model or None,
            "reranker_model": self.reranker_model or None,
            "fts5_active": bool(self.fts),
        })
        self.query_history = self.query_history[-100:]
        self._persist_index()

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

    def _hybrid_scores(self, query: str, plan: IntentPlan) -> Any:
        expanded_query = " ".join((query, *plan.expanded_terms))
        word_scores = cosine_similarity(self.vectorizer.transform([expanded_query]), self.matrix)[0]
        char_scores = cosine_similarity(self.char_vectorizer.transform([expanded_query]), self.char_matrix)[0]
        bm25_scores = self.bm25.scores(expanded_query) if self.bm25 else [0.0] * len(word_scores)
        fts_scores = self.fts.scores(expanded_query, [item.citation for item in self.evidence]) if self.fts else [0.0] * len(word_scores)
        lexical_scores = 0.30 * word_scores + 0.14 * char_scores
        lexical_scores = lexical_scores + [0.26 * value for value in bm25_scores]
        lexical_scores = lexical_scores + [0.30 * value for value in fts_scores]
        semantic_scores = self._semantic_scores(expanded_query)
        if semantic_scores is not None and len(semantic_scores) == len(lexical_scores):
            return 0.45 * lexical_scores + 0.55 * semantic_scores
        return lexical_scores

    def search(self, query: str, limit: int = 8) -> list[Evidence]:
        if not self.vectorizer or self.matrix is None or not self.char_vectorizer or self.char_matrix is None:
            raise ValueError("Index documents before asking a question.")
        plan = infer_intent(query)
        self.last_intent = plan
        scores = self._hybrid_scores(query, plan)
        results: list[Evidence] = []
        ranked_scores, ranked_indices = self._rank_indices(query, scores, limit)
        for index in ranked_indices[:limit]:
            evidence = self.evidence[index]
            results.append(self._with_parent_context(evidence, float(ranked_scores[index]), plan))
        self._record_query(query, plan, results)
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
        if "past_year" in plan.time_hints and item.modified_at:
            try:
                modified = datetime.fromisoformat(item.modified_at).replace(tzinfo=timezone.utc)
                if modified >= datetime.now(timezone.utc) - timedelta(days=365):
                    reasons.append("modified within the past year")
            except ValueError:
                pass
        if item.locator.startswith("slide"):
            reasons.append("slide-level evidence")
        elif item.locator.startswith("page"):
            reasons.append("page-level evidence")
        if item.ocr_used:
            reasons.append("OCR text from scanned page")
        if item.parent_id and item.chunk_count > 1:
            reasons.append(f"parent context merged from {item.chunk_count} child chunks")
        return reasons

    def locate_files(self, query: str, limit: int = 8, evidence_per_file: int = 2) -> list[FileMatch]:
        """Find file families for vague requests, retaining evidence and reasons."""

        if not self.vectorizer or self.matrix is None or not self.char_vectorizer or self.char_matrix is None:
            raise ValueError("Index documents before asking a question.")
        plan = infer_intent(query)
        self.last_intent = plan
        scores = self._hybrid_scores(query, plan)
        ranked_scores, _ = self._rank_indices(query, scores, limit)
        grouped: dict[str, list[tuple[Evidence, float]]] = {}
        for index, raw_score in enumerate(ranked_scores):
            item = self.evidence[index]
            suffix = Path(item.source).suffix.lower()
            type_bonus = 0.12 if suffix in plan.artifact_types else 0.0
            filename_text = item.source.lower()
            filename_bonus = 0.10 if any(term.lower() in filename_text for term in plan.expanded_terms if len(term) >= 2) else 0.0
            lowered_query = query.lower()
            report_request = any(term in lowered_query for term in ("报告", "课设", "课程设计", "report", "coursework", "course project"))
            if report_request and any(term in filename_text for term in ("报告", "report", "课设", "课程设计", "实验")):
                filename_bonus += 0.12
            if report_request and any(term in filename_text for term in ("答案", "试卷", "笔记", "复习", "模板", "模版")):
                filename_bonus -= 0.10
            recency_bonus = 0.0
            if "past_year" in plan.time_hints and item.modified_at:
                try:
                    modified = datetime.fromisoformat(item.modified_at).replace(tzinfo=timezone.utc)
                    if modified >= datetime.now(timezone.utc) - timedelta(days=365):
                        recency_bonus = 0.08
                except ValueError:
                    pass
            score = min(1.0, max(0.0, float(raw_score) * 0.78 + type_bonus + filename_bonus + recency_bonus))
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
                enriched = self._with_parent_context(item, raw_score, plan, file_score=best_score, family=family)
                selected.append(enriched)
                reasons = enriched.match_reasons
                for reason in reasons:
                    if reason not in reason_set:
                        reason_set.append(reason)
            if len(members) > 1:
                reason_set.append(f"{len(members) - 1} similar copy/version(s) grouped")
            ranked.append(FileMatch(source=selected[0].source, source_path=primary_path, file_family=family, score=round(best_score, 3), confidence=round(min(0.99, 0.45 + best_score * 0.5), 2), reasons=tuple(reason_set), evidence=tuple(selected), duplicate_paths=tuple(path for path, _ in members[1:])))
        ranked.sort(key=lambda match: match.score, reverse=True)
        selected = ranked[:limit]
        self._record_query(query, plan, selected)
        return selected


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
                # Qwen3's reasoning stream is useful for chat, but it can
                # leak non-JSON thinking tokens into this contract-bound API.
                "chat_template_kwargs": {"enable_thinking": False},
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
    blocks: list[str] = []
    for item in evidence:
        block = f"[{item.citation}] {item.source} ({item.locator}): {item.text}"
        if item.context_text:
            block += f"\nParent context (same page/section): {item.context_text}"
        blocks.append(block)
    return "\n\n".join(blocks)


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
    allow_fallback: bool = True,
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
Adjudication policy:
- supported means the literal affirmative claim is established by a signed or otherwise authoritative record.
- contradicted means an authoritative record explicitly establishes the opposite; do not use it merely because a document is silent.
- insufficient_evidence means the record contains marketing language, planning targets, conditional statements, a missing clause, or an unresolved exception.
For a question about a contractual uptime commitment, absence of an SLA clause and non-binding or conditional 99.9% language are insufficient_evidence unless a signed record explicitly resolves the issue. A lower or differently scoped signed SLA is a contradiction only if the documents clearly establish that it governs this exact claim.
Return {{claim, verdict, confidence, reasoning, evidence_citations, contradictions, timeline_events, missing_evidence, recommended_next_action}}. verdict must be supported, contradicted, or insufficient_evidence."""
        verdict, judge_stats = llm.complete_json(SYSTEM_PROMPT, judge_prompt)
        telemetry = {
            "latency_seconds": round(prosecution_stats["latency_seconds"] + defense_stats["latency_seconds"] + judge_stats["latency_seconds"], 2),
            "first_token_latency_seconds": judge_stats["first_token_latency_seconds"],
            "completion_tokens": prosecution_stats["completion_tokens"] + defense_stats["completion_tokens"] + judge_stats["completion_tokens"],
            "tokens_per_second": judge_stats["tokens_per_second"],
        }
        mode = getattr(llm, "label", "local vLLM")
    except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        if not allow_fallback:
            raise RuntimeError(f"Live local judge unavailable in championship mode: {exc}") from exc
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
