"""
RAG Evaluation Framework
========================
Wraps RAGAS (https://docs.ragas.io) for automated, reference-free evaluation
of retrieval quality and answer faithfulness.

Metrics (all model-based, no labelled ground truth needed for LLM metrics)
---------------------------------------------------------------------------
  faithfulness          – Is the answer grounded in the retrieved context?
  answer_relevancy      – Is the answer on-topic for the question?
  context_precision     – Are the retrieved chunks relevant to the question?
  context_recall        – Does the retrieved context cover the ground truth? (needs GT)

Domain Gold Standard
--------------------
A curated set of 50 gene-editing Q&A pairs is bundled in
  data/eval/gold_standard.jsonl
Each line: {"question":…, "ground_truth":…, "reference_dois":[ … ]}

Run evaluation
--------------
  from core.evaluator import RAGEvaluator
  ev = RAGEvaluator()
  score = ev.evaluate_response(question, answer, context_chunks)
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Gold standard path
# ---------------------------------------------------------------------------
GOLD_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval", "gold_standard.jsonl")


# ---------------------------------------------------------------------------
# Lightweight heuristic metrics (no extra dependency)
# ---------------------------------------------------------------------------

def _token_overlap(a: str, b: str) -> float:
    """Jaccard token overlap as a cheap recall proxy."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _faithfulness_heuristic(answer: str, contexts: List[str]) -> float:
    """
    Heuristic faithfulness: fraction of answer sentences that share ≥1 token
    with at least one retrieved chunk (very rough but cheap).
    """
    import re
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", answer) if len(s.strip()) > 20]
    if not sentences:
        return 1.0
    combined_context = " ".join(contexts).lower()
    grounded = 0
    for sent in sentences:
        words = [w for w in sent.lower().split() if len(w) > 4]
        if any(w in combined_context for w in words):
            grounded += 1
    return grounded / len(sentences)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Evaluates RAG responses using RAGAS when available, or lightweight
    heuristics as fallback.

    Parameters
    ----------
    llm_client : LLMClient (optional) – used for LLM-based metrics
    use_ragas  : bool – if False, skip RAGAS even if installed
    """

    def __init__(
        self,
        llm_client: Optional["LLMClient"] = None,
        use_ragas: bool = True,
    ):
        self.llm = llm_client
        self.use_ragas = use_ragas
        self._ragas_available = False

        if use_ragas:
            try:
                import ragas  # noqa: F401
                self._ragas_available = True
                print("[Evaluator] RAGAS is available for LLM-based metrics.")
            except ImportError:
                print("[Evaluator] RAGAS not installed; using heuristic metrics.")

    # -------------------------------------------------------------------------
    def evaluate_response(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics for a single Q&A pair.

        Returns a dict of metric_name → score (0.0 – 1.0).
        """
        metrics: Dict[str, float] = {}

        # ── Heuristic metrics (always available) ────────────────────────────
        metrics["faithfulness_heuristic"] = _faithfulness_heuristic(answer, contexts)
        metrics["context_token_overlap"] = float(
            max(_token_overlap(question, ctx) for ctx in contexts) if contexts else 0.0
        )
        if ground_truth:
            metrics["answer_coverage"] = _token_overlap(answer, ground_truth)

        # ── RAGAS metrics (when installed + LLM available) ──────────────────
        if self._ragas_available and self.llm and self.llm.client:
            try:
                metrics.update(self._ragas_eval(question, answer, contexts, ground_truth))
            except Exception as exc:
                print(f"[Evaluator] RAGAS evaluation failed: {exc}")

        metrics["evaluated_at"] = time.time()
        return metrics

    # -------------------------------------------------------------------------
    def _ragas_eval(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str],
    ) -> Dict[str, float]:
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas.metrics import faithfulness, answer_relevancy, context_precision  # type: ignore

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)
        metrics_to_use = [faithfulness, answer_relevancy, context_precision]

        result = evaluate(dataset, metrics=metrics_to_use)
        return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}

    # -------------------------------------------------------------------------
    def evaluate_batch(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate a list of dicts, each with keys:
          question, answer, contexts (list[str]), ground_truth (optional)
        """
        results = []
        for pair in qa_pairs:
            scores = self.evaluate_response(
                question=pair["question"],
                answer=pair["answer"],
                contexts=pair.get("contexts", []),
                ground_truth=pair.get("ground_truth"),
            )
            results.append({**pair, "scores": scores})
        return results

    # -------------------------------------------------------------------------
    def load_gold_standard(self) -> List[Dict[str, Any]]:
        """Load the bundled gene-editing gold-standard Q&A set."""
        if not os.path.exists(GOLD_PATH):
            print(f"[Evaluator] Gold standard not found at {GOLD_PATH}.")
            return []
        pairs = []
        try:
            with open(GOLD_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        pairs.append(json.loads(line))
            print(f"[Evaluator] Loaded {len(pairs)} gold-standard pairs.")
        except Exception as exc:
            print(f"[Evaluator] Error loading gold standard: {exc}")
        return pairs

    # -------------------------------------------------------------------------
    def run_gold_standard_eval(self) -> Dict[str, float]:
        """Run evaluation on the built-in gold standard set."""
        pairs = self.load_gold_standard()
        if not pairs:
            return {}
        results = self.evaluate_batch(pairs)
        # Average all numeric scores
        all_keys = set()
        for r in results:
            all_keys.update(r.get("scores", {}).keys())

        avg: Dict[str, float] = {}
        for k in all_keys:
            if k == "evaluated_at":
                continue
            vals = [r["scores"][k] for r in results if k in r.get("scores", {})]
            avg[k] = round(sum(vals) / len(vals), 4) if vals else 0.0

        print(f"[Evaluator] Gold Standard Results: {avg}")
        return avg
