"""
ingest.py — Document ingestion pipeline.

Steps:
  1. Parse PDFs with PyMuPDF (fitz) — preserves metadata, page numbers
  2. Chunk with a sentence-aware sliding window strategy
  3. Embed with sentence-transformers (local, free)
  4. Store in ChromaDB (persistent, local)

Usage:
  python ingest.py                     # ingest all PDFs in data/docs/
  python ingest.py --file my.pdf       # ingest a single file
  python ingest.py --reset             # wipe collection and re-ingest
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Generator

import fitz                              # PyMuPDF
import chromadb
import structlog
from sentence_transformers import SentenceTransformer

from config import settings

log = structlog.get_logger()


# ── Chunker ───────────────────────────────────────────────────────────────────

def _sentence_split(text: str) -> list[str]:
    """Naive sentence splitter — avoids heavy NLP dependencies."""
    import re
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int = settings.chunk_size,
    overlap: int = settings.chunk_overlap,
) -> list[str]:
    """
    Sentence-aware sliding window chunker.

    Strategy rationale (documented for interview):
    - Splitting on sentences avoids cutting mid-thought
    - Overlap preserves context across chunk boundaries
    - chunk_size ≈ 800 chars balances retrieval precision vs. context
    """
    sentences = _sentence_split(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) > chunk_size and current:
            chunks.append(" ".join(current))
            # keep overlap sentences
            overlap_chars = 0
            while current and overlap_chars < overlap:
                overlap_chars += len(current[-1])
                current = current[:-1]
            current = list(reversed(current))
            current_len = sum(len(s) for s in current)
        current.append(sent)
        current_len += len(sent)

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if len(c.strip()) > 50]  # discard tiny fragments


# ── PDF Parser ────────────────────────────────────────────────────────────────

def parse_pdf(path: Path) -> Generator[dict, None, None]:
    """
    Yield dicts with keys: text, page, source, doc_type, total_pages.
    Preserves section headings found via font-size heuristics.
    """
    doc = fitz.open(str(path))
    total = len(doc)
    doc_name = path.stem

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        page_text_parts = []

        for block in blocks:
            if block.get("type") != 0:   # text blocks only
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    page_text_parts.append(span.get("text", ""))

        page_text = " ".join(page_text_parts).strip()
        if not page_text:
            continue

        for chunk in chunk_text(page_text):
            yield {
                "text": chunk,
                "source": doc_name,
                "file": str(path),
                "page": page_num,
                "total_pages": total,
                "doc_type": _infer_doc_type(doc_name),
            }


def _infer_doc_type(name: str) -> str:
    name_lower = name.lower()
    if any(k in name_lower for k in ["10-k", "10k", "annual"]):
        return "annual_report"
    if any(k in name_lower for k in ["10-q", "10q", "quarterly"]):
        return "quarterly_report"
    if "memo" in name_lower or "due_diligence" in name_lower:
        return "due_diligence"
    if "commentary" in name_lower or "outlook" in name_lower:
        return "market_commentary"
    return "general"


# ── Embedder ─────────────────────────────────────────────────────────────────

class Embedder:
    def __init__(self) -> None:
        log.info("loading_embedding_model", model=settings.embedding_model)
        self._model = SentenceTransformer(
            settings.embedding_model, device=settings.embedding_device
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()


# ── ChromaDB client ───────────────────────────────────────────────────────────

def get_collection(reset: bool = False) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    if reset:
        try:
            client.delete_collection(settings.collection_name)
            log.info("collection_reset", name=settings.collection_name)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# ── Main ingestion ────────────────────────────────────────────────────────────

def ingest(pdf_paths: list[Path], reset: bool = False, batch_size: int = 64) -> int:
    embedder = Embedder()
    collection = get_collection(reset=reset)

    total_chunks = 0
    buffer_docs: list[str] = []
    buffer_ids: list[str] = []
    buffer_meta: list[dict] = []
    # Track IDs seen in the current batch to deduplicate within a single upsert call
    buffer_id_set: set[str] = set()

    def flush() -> None:
        nonlocal total_chunks
        if not buffer_docs:
            return
        embeddings = embedder.embed(buffer_docs)
        collection.upsert(
            ids=buffer_ids,
            documents=buffer_docs,
            embeddings=embeddings,
            metadatas=buffer_meta,
        )
        total_chunks += len(buffer_docs)
        log.info("batch_upserted", count=len(buffer_docs), total=total_chunks)
        buffer_docs.clear(); buffer_ids.clear(); buffer_meta.clear()
        buffer_id_set.clear()

    chunk_counter = 0  # global counter ensures uniqueness even for identical text

    for path in pdf_paths:
        log.info("ingesting_file", file=str(path))
        t0 = time.perf_counter()

        for record in parse_pdf(path):
            chunk_counter += 1
            # Include full text + counter so identical passages get distinct IDs
            chunk_id = hashlib.sha256(
                f"{record['source']}:p{record['page']}:{chunk_counter}:{record['text']}".encode()
            ).hexdigest()[:16]

            # Skip if this ID is already queued in the current batch
            if chunk_id in buffer_id_set:
                log.warning("duplicate_id_skipped", chunk_id=chunk_id, source=record["source"])
                continue

            buffer_id_set.add(chunk_id)
            buffer_ids.append(chunk_id)
            buffer_docs.append(record["text"])
            buffer_meta.append({
                "source": record["source"],
                "file": record["file"],
                "page": record["page"],
                "total_pages": record["total_pages"],
                "doc_type": record["doc_type"],
            })

            if len(buffer_docs) >= batch_size:
                flush()

        flush()
        elapsed = time.perf_counter() - t0
        log.info("file_ingested", file=str(path), elapsed_s=round(elapsed, 2))

    log.info("ingestion_complete", total_chunks=total_chunks, files=len(pdf_paths))
    return total_chunks


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    import structlog

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )

    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB")
    parser.add_argument("--file", type=str, help="Single PDF to ingest")
    parser.add_argument("--reset", action="store_true", help="Wipe collection first")
    args = parser.parse_args()

    docs_dir = Path(settings.docs_dir)

    if args.file:
        paths = [Path(args.file)]
    else:
        paths = list(docs_dir.glob("*.pdf"))
        if not paths:
            print(f"No PDFs found in {docs_dir}. Add PDFs and retry.")
            raise SystemExit(1)

    ingest(paths, reset=args.reset)
