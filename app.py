"""
app.py — Streamlit UI for the Document Intelligence PoC.

Run: streamlit run app.py

Features:
  - Upload & ingest PDFs directly from the browser
  - Ask natural-language questions
  - View structured response: answer, citations, confidence, human-review flag
  - Expandable debug panel: retrieved chunks, query type, latency, token cost
  - Observability sidebar: per-session stats (queries, avg latency, review flags)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st
import structlog

from config import settings
from ingest import ingest
from rag import ask, DocumentResponse

log = structlog.get_logger()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session state defaults ────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "history": [],        # list of (question, DocumentResponse)
        "total_queries": 0,
        "total_tokens": 0,
        "total_latency_ms": 0.0,
        "review_flags": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Document Intelligence")
    st.caption("Document Intelligence — PoC")
    st.divider()

    # Observability stats
    st.subheader("Session Stats")
    col1, col2 = st.columns(2)
    n = st.session_state.total_queries or 1
    col1.metric("Queries", st.session_state.total_queries)
    col2.metric("Review Flags", st.session_state.review_flags)
    avg_lat = (st.session_state.total_latency_ms / n) if st.session_state.total_queries else 0
    st.metric("Avg Latency", f"{avg_lat:.0f} ms")
    st.metric("Tokens Used", st.session_state.total_tokens)
    st.caption("Cost: $0.00 (local Ollama)")

    st.divider()

    # Document ingestion
    st.subheader("📄 Ingest Documents")
    uploaded = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True
    )
    reset_col = st.checkbox("Reset collection before ingesting", value=False)

    if st.button("Ingest", disabled=not uploaded):
        docs_dir = Path(settings.docs_dir)
        docs_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for uf in uploaded:
            dest = docs_dir / uf.name
            dest.write_bytes(uf.read())
            saved.append(dest)
        with st.spinner(f"Ingesting {len(saved)} document(s)…"):
            n_chunks = ingest(saved, reset=reset_col)
        st.success(f"✅ {n_chunks} chunks indexed from {len(saved)} file(s)")

    st.divider()

    # Model info
    st.subheader("⚙️ Config")
    st.code(
        f"LLM:        {settings.llm_model}\n"
        f"Embeddings: {settings.embedding_model}\n"
        f"Vector DB:  ChromaDB (local)\n"
        f"Retrieval:  Hybrid BM25 + Semantic\n"
        f"top_k:      {settings.top_k}",
        language=None,
    )


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Ask Your Financial Documents")
st.caption(
    "Powered by Ollama (local LLM) · sentence-transformers · ChromaDB · LangChain"
)


def _display_response(response: DocumentResponse) -> None:
    """Render a DocumentResponse in the chat message area."""
    # Confidence badge
    badge_map = {"high": "🟢", "medium": "🟡", "low": "🔴"}
    badge = badge_map.get(response.confidence, "⚪")
    st.write(response.answer)

    cols = st.columns(4)
    cols[0].caption(f"{badge} Confidence: **{response.confidence}**")
    cols[1].caption(f"🏷 Type: **{response.query_type}**")
    cols[2].caption(f"⏱ {response.latency_ms:.0f} ms")
    cols[3].caption(f"🔢 {response.tokens_used} tokens")

    if response.requires_human_review:
        st.warning("⚠️ **Human review recommended.** "
                   + (response.uncertainty or "Low confidence or unverified citations."))

    if response.citations:
        with st.expander(f"📎 Citations ({len(response.citations)})"):
            for i, cit in enumerate(response.citations, 1):
                verified_icon = "✅" if cit.verified else "❌"
                st.markdown(
                    f"**[{i}] {verified_icon} {cit.source}** — Page {cit.page}\n\n"
                    f"> _{cit.quote}_"
                )


# ── Conversation history display ──────────────────────────────────────────────
for q, r in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        _display_response(r)

# ── Query input ───────────────────────────────────────────────────────────────
question = st.chat_input("Ask a question about your financial documents…")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Classifying query · Retrieving chunks · Generating answer…"):
            t0 = time.perf_counter()
            response = ask(question)
            wall_ms = (time.perf_counter() - t0) * 1000

        _display_response(response)

        # Update session stats
        st.session_state.total_queries += 1
        st.session_state.total_tokens += response.tokens_used
        st.session_state.total_latency_ms += wall_ms
        if response.requires_human_review:
            st.session_state.review_flags += 1

        st.session_state.history.append((question, response))
        log.info("ui_query", question=question[:80], latency_ms=round(wall_ms, 1))
