# Document Intelligence — Simple RAG PoC

A lightweight, fully open-source RAG assistant for financial documents.

**Stack: all free, all local** — Ollama · sentence-transformers · ChromaDB · Streamlit · MLflow

---

## Architecture

```
PDF Documents
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  INGESTION PIPELINE (ingest.py)                     │
│  PyMuPDF → sentence-aware chunker → SentenceTransformers │
│                         │                           │
│                         ▼                           │
│                   ChromaDB (local)                  │
└─────────────────────────────────────────────────────┘
                          │
                    User Query
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│  AGENTIC ROUTER (rag.py)                            │
│  classify_query → factual / comparison / summary /  │
│                   temporal / out_of_scope           │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│  HYBRID RETRIEVER                                   │
│  BM25 (keyword, 30%) + Semantic (cosine, 70%)       │
│  → top-K chunks with metadata                      │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│  GENERATOR (Ollama / Llama 3.1 8B)                  │
│  Structured prompt → JSON response                  │
│  CitationVerifier → hallucination guard             │
│  Pydantic model → DocumentResponse                  │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
              Streamlit UI + MLflow observability
```

---

## Quick Start

### 1. Prerequisites

```bash
# Install Ollama (free, local LLM runtime)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b        # ~4.7 GB download

# Install Python deps
pip install -r requirements.txt
```

### 2. Add documents

```bash
mkdir -p data/docs
# Copy your PDFs into data/docs/
cp my_fund_report.pdf data/docs/
```

### 3. Ingest

```bash
python ingest.py               # ingest all PDFs in data/docs/
python ingest.py --reset       # wipe and re-ingest
python ingest.py --file x.pdf  # single file
```

### 4. Run the UI

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### 5. Evaluate

```bash
# Edit data/gold_set.json with questions from your corpus first
python evaluate.py
# Report saved to data/eval_results.json
```

### 6. View observability (MLflow)

```bash
mlflow ui --backend-store-uri ./data/mlflow
# Opens at http://localhost:5000
```

---

## Design Decisions & Trade-offs

### Embedding model: `all-MiniLM-L6-v2`
- **Why:** 80MB, runs on CPU, ~22ms/query, strong retrieval performance on financial text
- **Trade-off:** Larger models (e.g. `bge-large-en-v1.5`) score ~3-5% higher on BEIR but are 5× slower
- **Azure alternative:** Azure AI Foundry text-embedding-3-small ($0.00002/1K tokens)

### Vector store: ChromaDB (local)
- **Why:** Zero-config, persistent, free, cosine similarity built in
- **Trade-off:** Not horizontally scalable; at >1M chunks switch to Azure AI Search or Weaviate
- **Scale threshold:** ~50K chunks before query latency exceeds 500ms on CPU

### Hybrid retrieval: BM25 (30%) + Semantic (70%)
- **Why:** BM25 excels on fund names, tickers, exact numbers; semantic handles paraphrase
- **Trade-off:** Two rankings to merge (RRF or weighted); adds ~20ms vs pure semantic
- **Tuning:** Adjust `bm25_weight` / `semantic_weight` in config.py

### LLM: Ollama (Llama 3.1 8B)
- **Why:** Completely free, no API key, runs locally, no data leaves the machine
- **Trade-off:** ~2-4s latency vs ~0.5s for GPT-4o API; quality slightly lower on complex reasoning
- **Swap:** Change `ollama_base_url` + `llm_model` in `.env` for Azure OpenAI or Databricks

### Agentic routing: zero-shot prompt classifier
- **Why:** Simple, cheap, interpretable — one LLM call to classify before retrieval
- **Trade-off:** ~500ms overhead; a fine-tuned SetFit classifier would be faster (<10ms)
- **Improvement path:** Replace with embedding-based few-shot classifier at scale

### Structured output: Pydantic `DocumentResponse`
- **Why:** Forces the LLM into a reliable schema; easy to validate, log, and route
- **Trade-off:** Prompt engineering overhead; occasionally the model breaks JSON format
- **Mitigation:** Strip code fences, retry on parse failure (not yet implemented — see Next Steps)

---

## Evaluation Results

Run `python evaluate.py` after ingesting your corpus and editing `data/gold_set.json`.

Expected metrics with a good corpus and `llama3.1:8b`:

| Metric | Target | Notes |
|--------|--------|-------|
| Mean accuracy (LLM-judge) | ≥ 0.70 | Depends heavily on document quality |
| Citation precision | ≥ 0.80 | % of citations verified in retrieved chunks |
| Hallucination rate | ≤ 0.20 | Unverified citations |
| p50 latency | < 4s | On CPU with 8B model |
| Out-of-scope refusal | ≥ 0.90 | Boundary test accuracy |

---

## Next Steps (if this were going to production)

1. **Retry on JSON parse failure** — exponential backoff + simplified prompt
2. **Reranking** — add a cross-encoder reranker (e.g. `ms-marco-MiniLM-L-6-v2`) before generation
3. **Query decomposition** — break complex comparisons into sub-queries
4. **Caching** — semantic cache for repeated queries (LangChain `GPTCache` or Redis)
5. **Azure AI Search** — swap ChromaDB for production-grade vector search with RBAC
6. **Human-in-the-loop** — route `requires_human_review=True` responses to a review queue
7. **Drift monitoring** — track retrieval quality over time as documents are added
8. **Auth** — add Streamlit authentication; restrict document access by user role

---

## Governance & Responsible AI

- **Prompt injection:** User queries are separated from document context in the prompt template. Input length is not validated yet (Next Steps).
- **PII:** No PII is logged. Query text is logged at INFO level — review before production.
- **Data residency:** All data stays local. No external API calls unless you configure Azure OpenAI.
- **Hallucination guard:** `CitationVerifier` flags any citation where the quote is not found in retrieved chunks.
- **Human review flag:** `requires_human_review=True` is set when confidence is low or citations are unverified.

---

## Repo Structure

```
simple_RAG/
├── config.py          # All settings (env vars / .env)
├── ingest.py          # PDF → chunks → embeddings → ChromaDB
├── rag.py             # Query router → retriever → generator → structured output
├── evaluate.py        # Gold set evaluation runner
├── app.py             # Streamlit UI
├── requirements.txt
├── .env.example       # Copy to .env and configure
├── data/
│   ├── docs/          # Put your PDFs here
│   ├── chroma/        # ChromaDB persists here
│   ├── mlflow/        # MLflow tracking store
│   └── gold_set.json  # Evaluation Q&A pairs
└── .github/
    └── workflows/
        └── evaluate.yml  # CI evaluation pipeline
```
