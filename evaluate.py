"""
evaluate.py — Gold set evaluation runner.

Measures:
  - Answer accuracy (LLM-as-judge 0-1)
  - Citation precision (% of citations verified in retrieved chunks)
  - Hallucination rate (% of unverified citations)
  - Latency (p50, p99 in ms)
  - Out-of-scope refusal accuracy

Usage:
  python evaluate.py
  python evaluate.py --gold data/gold_set.json --report results.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Optional

import requests
import structlog

from config import settings
from rag import ask, DocumentResponse

log = structlog.get_logger()


# ── LLM-as-judge ──────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are evaluating a Q&A system for financial documents.

Question: {question}
Expected answer (ground truth): {expected}
System answer: {actual}

Score the system answer on accuracy from 0 to 1:
  1.0 = fully correct and complete
  0.5 = partially correct, minor omission or imprecision
  0.0 = wrong, hallucinated, or refused when it should not have

Respond ONLY with a JSON object: {{"score": <0-1>, "reason": "<one sentence>"}}
"""


def llm_judge(question: str, expected: str, actual: str) -> tuple[float, str]:
    """Use the local LLM to score answer quality. Returns (score, reason)."""
    prompt = JUDGE_PROMPT.format(question=question, expected=expected, actual=actual)
    try:
        resp = requests.post(
            f"{settings.ollama_base_url}/api/generate",
            json={"model": settings.llm_model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0, "num_predict": 100}},
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "{}").strip().strip("```json").strip("```")
        parsed = json.loads(raw)
        return float(parsed.get("score", 0.0)), parsed.get("reason", "")
    except Exception as exc:
        log.warning("judge_error", error=str(exc))
        return 0.0, f"judge error: {exc}"


# ── Metrics ───────────────────────────────────────────────────────────────────

def citation_precision(response: DocumentResponse) -> float:
    """% of citations where the quote was verified in retrieved chunks."""
    if not response.citations:
        return 1.0  # no citations to verify
    verified = sum(1 for c in response.citations if c.verified)
    return verified / len(response.citations)


def hallucination_rate(response: DocumentResponse) -> float:
    return 1.0 - citation_precision(response)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_evaluation(gold_path: str = settings.gold_set_path) -> dict:
    with open(gold_path) as f:
        gold_set = json.load(f)

    results = []
    latencies: list[float] = []
    accuracy_scores: list[float] = []
    cit_precisions: list[float] = []
    refusal_correct = 0
    refusal_total = 0

    print(f"\n{'='*60}")
    print(f"  Pantheon RAG — Evaluation ({len(gold_set)} questions)")
    print(f"{'='*60}\n")

    for i, item in enumerate(gold_set, 1):
        question = item["question"]
        expected = item["expected_answer"]
        expected_type = item.get("expected_query_type", "factual")
        is_refusal = expected_type == "out_of_scope"

        print(f"[{i}/{len(gold_set)}] {question[:70]}")

        t0 = time.perf_counter()
        response = ask(question)
        wall_ms = (time.perf_counter() - t0) * 1000

        # Accuracy via LLM judge
        if is_refusal:
            refusal_total += 1
            refused = response.query_type == "out_of_scope" or "outside" in response.answer.lower()
            score = 1.0 if refused else 0.0
            reason = "correctly refused" if refused else "should have refused"
            if refused:
                refusal_correct += 1
        else:
            score, reason = llm_judge(question, expected, response.answer)

        cit_prec = citation_precision(response)
        latencies.append(wall_ms)
        accuracy_scores.append(score)
        cit_precisions.append(cit_prec)

        result = {
            "question": question,
            "expected_type": expected_type,
            "actual_type": response.query_type,
            "accuracy_score": round(score, 3),
            "judge_reason": reason,
            "citation_precision": round(cit_prec, 3),
            "hallucination_rate": round(1.0 - cit_prec, 3),
            "latency_ms": round(wall_ms, 1),
            "confidence": response.confidence,
            "requires_human_review": response.requires_human_review,
            "tokens_used": response.tokens_used,
            "answer_preview": response.answer[:120],
        }
        results.append(result)
        print(f"  accuracy={score:.2f}  cit_prec={cit_prec:.2f}  latency={wall_ms:.0f}ms  [{reason[:50]}]")

    # ── Aggregate report ──────────────────────────────────────────────────
    lat_sorted = sorted(latencies)
    p50 = lat_sorted[len(lat_sorted) // 2]
    p99 = lat_sorted[int(len(lat_sorted) * 0.99)]

    summary = {
        "total_questions": len(gold_set),
        "mean_accuracy": round(statistics.mean(accuracy_scores), 3),
        "mean_citation_precision": round(statistics.mean(cit_precisions), 3),
        "mean_hallucination_rate": round(1.0 - statistics.mean(cit_precisions), 3),
        "refusal_accuracy": round(refusal_correct / refusal_total, 3) if refusal_total else None,
        "latency_p50_ms": round(p50, 1),
        "latency_p99_ms": round(p99, 1),
        "requires_human_review_count": sum(1 for r in results if r["requires_human_review"]),
    }

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for k, v in summary.items():
        print(f"  {k:<35} {v}")
    print(f"{'='*60}\n")

    return {"summary": summary, "results": results}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    import structlog as sl

    sl.configure(wrapper_class=sl.make_filtering_bound_logger(logging.INFO))

    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=settings.gold_set_path)
    parser.add_argument("--report", default="./data/eval_results.json")
    args = parser.parse_args()

    report = run_evaluation(args.gold)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full report saved to {args.report}")
