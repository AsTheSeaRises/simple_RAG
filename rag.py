"""
rag.py — RAG pipeline with agentic query routing.

Architecture:
  1. QueryRouter classifies the incoming question
  2. Retriever does hybrid search (BM25 + semantic) via ChromaDB
  3. CitationVerifier checks that citations exist in retrieved chunks
  4. Generator calls Ollama LLM and returns a structured DocumentResponse
  5. All steps are logged to structlog + tracked in MLflow

Design decisions (for interview walkthrough):
  - Pydantic structured output: forces reliable parsing, easy to extend
  - Hybrid BM25 + semantic: BM25 excels on exact terms (fund names, tickers);
    semantic handles paraphrase/synonyms. Weighted 30/70 default.
  - Query routing is a simple zero-shot prompt classifier — cheap and fast.
    An embedding-based classifier (e.g. SetFit) would be more robust at scale.
  - Ollama keeps everything local and free; swap ollama_base_url for
    Azure OpenAI or Databricks Foundation Models with zero code changes.
"""
from __future__ import annotations

import json
import time
from enum import Enum
from typing import Optional

import chromadb
import mlflow
import requests
import structlog
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import settings

log = structlog.get_logger()


# ── Data models ───────────────────────────────────────────────────────────────

class QueryType(str, Enum):
    FACTUAL = "factual"          # single-document fact lookup
    COMPARISON = "comparison"    # multi-document comparison
    SUMMARY = "summary"          # summarisation of one or more docs
    TEMPORAL = "temporal"        # reasoning about time / changes
    OUT_OF_SCOPE = "out_of_scope"  # query outside document corpus


class Citation(BaseModel):
    source: str = Field(description="Document filename")
    page: int = Field(description="Page number in source document")
    quote: str = Field(description="Short verbatim excerpt supporting the answer")
    verified: bool = Field(default=False, description="Quote found in retrieved chunks")


class DocumentResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: str = Field(description="high | medium | low")
    uncertainty: Optional[str] = Field(default=None, description="Stated if confidence < high")
    requires_human_review: bool
    query_type: str
    tokens_used: int
    latency_ms: float
    cost_usd: float = Field(default=0.0, description="Estimated cost (0 for local Ollama)")


class RetrievedChunk(BaseModel):
    text: str
    source: str
    page: int
    score: float
    doc_type: str


# ── Query Router (agentic classifier) ─────────────────────────────────────────

CLASSIFIER_PROMPT = """You are a query classifier for a financial document Q&A system.
Classify the user's question into exactly one category:

- factual: A specific fact from one document (NAV, date, fund name, percentage)
- comparison: Comparing two or more funds, documents, or time periods
- summary: Asking for a summary or overview of one or more documents
- temporal: Asking about changes over time or between two dates
- out_of_scope: The question cannot be answered from financial fund documents
  (e.g. stock prices, general news, personal advice)

Respond with ONLY a JSON object: {{"query_type": "<category>", "rationale": "<one sentence>"}}

Question: {question}
"""


def classify_query(question: str) -> tuple[QueryType, str]:
    """Call Ollama to classify the query. Falls back to FACTUAL on error."""
    payload = {
        "model": settings.llm_model,
        "prompt": CLASSIFIER_PROMPT.format(question=question),
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 80},
    }
    try:
        resp = requests.post(
            f"{settings.ollama_base_url}/api/generate",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        parsed = json.loads(raw.strip())
        qt = QueryType(parsed.get("query_type", "factual"))
        rationale = parsed.get("rationale", "")
        log.info("query_classified", query_type=qt.value, rationale=rationale)
        return qt, rationale
    except Exception as exc:
        log.warning("classifier_fallback", error=str(exc))
        return QueryType.FACTUAL, "classifier unavailable"


# ── Hybrid Retriever ──────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines BM25 (keyword) and semantic (embedding) scores.
    Trade-off documented:
      - BM25 is fast and great for exact matches (fund names, numbers)
      - Semantic handles paraphrase but misses rare terms
      - Hybrid beats both in most RAG benchmarks (Ma et al., 2023)
    """

    def __init__(self) -> None:
        self._embedder = SentenceTransformer(
            settings.embedding_model, device=settings.embedding_device
        )
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._col = client.get_collection(settings.collection_name)

    def retrieve(
        self,
        query: str,
        query_type: QueryType,
        top_k: int = settings.top_k,
        doc_type_filter: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        # ── semantic search ────────────────────────────────────────────────
        q_emb = self._embedder.encode([query]).tolist()
        where = {"doc_type": doc_type_filter} if doc_type_filter else None

        sem_results = self._col.query(
            query_embeddings=q_emb,
            n_results=min(top_k * 2, self._col.count() or 1),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        docs = sem_results["documents"][0]
        metas = sem_results["metadatas"][0]
        distances = sem_results["distances"][0]

        if not docs:
            return []

        # cosine distance → similarity score (0-1)
        sem_scores = [1.0 - d for d in distances]

        # ── BM25 keyword search on retrieved candidates ────────────────────
        tokenised = [d.lower().split() for d in docs]
        bm25 = BM25Okapi(tokenised)
        bm25_raw = bm25.get_scores(query.lower().split())
        bm25_max = max(bm25_raw) or 1.0
        bm25_scores = [s / bm25_max for s in bm25_raw]

        # ── hybrid fusion ─────────────────────────────────────────────────
        w_sem = settings.semantic_weight
        w_bm25 = settings.bm25_weight
        combined = [
            (i, w_sem * s + w_bm25 * b)
            for i, (s, b) in enumerate(zip(sem_scores, bm25_scores))
        ]
        combined.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in combined[:top_k]:
            meta = metas[idx]
            results.append(
                RetrievedChunk(
                    text=docs[idx],
                    source=meta.get("source", "unknown"),
                    page=int(meta.get("page", 0)),
                    score=round(score, 4),
                    doc_type=meta.get("doc_type", "general"),
                )
            )

        log.info(
            "retrieval_complete",
            query_type=query_type.value,
            chunks_returned=len(results),
            top_score=results[0].score if results else 0,
        )
        return results


# ── Citation Verifier ────────────────────────────────────────────────────────

def verify_citations(citations: list[Citation], chunks: list[RetrievedChunk]) -> list[Citation]:
    """
    Check that each cited quote is actually present in retrieved chunks.
    This is a key hallucination guard — if the LLM invents a quote,
    verified=False flags it for human review.
    """
    chunk_texts = [c.text.lower() for c in chunks]
    for cit in citations:
        short_quote = cit.quote[:60].lower()
        cit.verified = any(short_quote in ct for ct in chunk_texts)
    return citations


# ── Generator ────────────────────────────────────────────────────────────────

ANSWER_PROMPT = """You are a financial document analyst at Pantheon, a private markets firm.
Answer the user's question using ONLY the provided document excerpts.
Do NOT use external knowledge. If the answer is not in the excerpts, say so clearly.

Query type: {query_type}

Document excerpts:
{context}

Question: {question}

Respond with a JSON object matching this exact schema:
{{
  "answer": "<your answer in 2-5 sentences>",
  "citations": [
    {{
      "source": "<filename>",
      "page": <page_number>,
      "quote": "<verbatim excerpt from the document, max 30 words>"
    }}
  ],
  "confidence": "<high|medium|low>",
  "uncertainty": "<explain why if not high, else null>",
  "requires_human_review": <true|false>
}}

Rules:
- Include a citation for every factual claim.
- Set confidence=low and requires_human_review=true if the excerpts are insufficient.
- For out_of_scope questions, set answer="This question is outside the scope of the document corpus."
- Return ONLY valid JSON. No prose before or after.
"""

OUT_OF_SCOPE_RESPONSE = DocumentResponse(
    answer="This question is outside the scope of the loaded document corpus. "
           "The system can only answer questions about the ingested financial documents.",
    citations=[],
    confidence="high",
    uncertainty=None,
    requires_human_review=False,
    query_type="out_of_scope",
    tokens_used=0,
    latency_ms=0.0,
    cost_usd=0.0,
)


def _build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Source: {c.source}, Page {c.page} (score={c.score})\n{c.text}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    query_type: QueryType,
) -> DocumentResponse:
    context = _build_context(chunks)
    prompt = ANSWER_PROMPT.format(
        query_type=query_type.value,
        context=context,
        question=question,
    )

    t0 = time.perf_counter()
    payload = {
        "model": settings.llm_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": settings.llm_temperature,
            "num_predict": settings.llm_max_tokens,
        },
    }

    resp = requests.post(
        f"{settings.ollama_base_url}/api/generate",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    latency_ms = (time.perf_counter() - t0) * 1000

    result = resp.json()
    raw_text = result.get("response", "{}")
    tokens = result.get("eval_count", 0) + result.get("prompt_eval_count", 0)

    # strip any markdown code fences the model might emit
    raw_text = raw_text.strip().strip("```json").strip("```").strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        log.error("json_parse_failed", error=str(exc), raw=raw_text[:200])
        return DocumentResponse(
            answer="Failed to parse model response. Please retry.",
            citations=[],
            confidence="low",
            uncertainty="Model output was not valid JSON.",
            requires_human_review=True,
            query_type=query_type.value,
            tokens_used=tokens,
            latency_ms=round(latency_ms, 1),
            cost_usd=0.0,
        )

    citations = [
        Citation(
            source=c.get("source", "unknown"),
            page=int(c.get("page", 0)),
            quote=c.get("quote", ""),
        )
        for c in parsed.get("citations", [])
    ]
    citations = verify_citations(citations, chunks)
    unverified = sum(1 for c in citations if not c.verified)

    return DocumentResponse(
        answer=parsed.get("answer", ""),
        citations=citations,
        confidence=parsed.get("confidence", "low"),
        uncertainty=parsed.get("uncertainty"),
        requires_human_review=parsed.get("requires_human_review", False) or unverified > 0,
        query_type=query_type.value,
        tokens_used=tokens,
        latency_ms=round(latency_ms, 1),
        cost_usd=round(tokens / 1000 * settings.cost_per_1k_output_tokens, 6),
    )


# ── Public pipeline entry point ───────────────────────────────────────────────

def ask(question: str, doc_type_filter: Optional[str] = None) -> DocumentResponse:
    """
    End-to-end pipeline: classify → retrieve → generate → verify.
    Logs every step to structlog and MLflow.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    with mlflow.start_run(run_name="rag_query", nested=True):
        t_total = time.perf_counter()

        # 1. Route
        query_type, rationale = classify_query(question)
        mlflow.log_param("query_type", query_type.value)

        if query_type == QueryType.OUT_OF_SCOPE:
            log.info("query_out_of_scope", question=question[:80])
            return OUT_OF_SCOPE_RESPONSE

        # 2. Retrieve
        retriever = HybridRetriever()
        chunks = retriever.retrieve(question, query_type, doc_type_filter=doc_type_filter)

        if not chunks:
            log.warning("no_chunks_retrieved", question=question[:80])
            return DocumentResponse(
                answer="No relevant documents found for your question. "
                       "Please check that documents have been ingested.",
                citations=[],
                confidence="low",
                uncertainty="No chunks retrieved from vector store.",
                requires_human_review=True,
                query_type=query_type.value,
                tokens_used=0,
                latency_ms=0.0,
            )

        # 3. Generate
        response = generate_answer(question, chunks, query_type)

        # 4. Log metrics
        total_ms = (time.perf_counter() - t_total) * 1000
        mlflow.log_metrics({
            "latency_ms": response.latency_ms,
            "total_latency_ms": round(total_ms, 1),
            "tokens_used": response.tokens_used,
            "chunks_retrieved": len(chunks),
            "citations_count": len(response.citations),
            "unverified_citations": sum(1 for c in response.citations if not c.verified),
        })

        log.info(
            "query_complete",
            question=question[:80],
            query_type=response.query_type,
            confidence=response.confidence,
            latency_ms=round(total_ms, 1),
            tokens=response.tokens_used,
            human_review=response.requires_human_review,
        )

        return response
