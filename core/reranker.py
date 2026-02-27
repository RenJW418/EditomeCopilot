"""
Cross-Encoder Re-Ranker
=======================
Replaces naive top-k retrieval with a two-stage pipeline:
  1. Bi-encoder FAISS retrieval (fast, coarse)
  2. Cross-encoder reranking (slow, precise)

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22 M params, runs on CPU in < 200 ms for 20 candidates
  - Trained on MS-MARCO passage relevance, generalises well to biomedical text
  - For heavier GPUs: cross-encoder/ms-marco-electra-base (better quality)
"""

from __future__ import annotations

import os
from typing import List, Tuple, Any

# Lazy import so that cold-start doesn't fail if sentence-transformers is absent
_CrossEncoder = None

def _get_cross_encoder(model_name: str):
    global _CrossEncoder
    if _CrossEncoder is None:
        from sentence_transformers import CrossEncoder as _CE
        _CrossEncoder = _CE
    return _CrossEncoder(model_name)


class CrossEncoderReranker:
    """
    Stateless re-ranker wrapping a HuggingFace Cross-Encoder.

    Parameters
    ----------
    model_name : str
        Any sentence-transformers cross-encoder model path or Hub ID.
    top_k : int
        How many candidates to return after reranking.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 20,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self._model = None
        print(f"[Reranker] Initialising cross-encoder: {model_name}")

    # -------------------------------------------------------------------------
    # Lazy model loader
    # -------------------------------------------------------------------------
    def _load(self):
        if self._model is None:
            try:
                self._model = _get_cross_encoder(self.model_name)
                print(f"[Reranker] Cross-encoder loaded: {self.model_name}")
            except Exception as exc:
                print(f"[Reranker] WARNING – could not load cross-encoder: {exc}")
                self._model = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Any, float]],
        top_k: int | None = None,
    ) -> List[Tuple[Any, float]]:
        """
        Re-rank a list of (Document, score) tuples.

        Parameters
        ----------
        query : str
        candidates : list of (LangChain Document, float)
        top_k : override instance-level top_k for this call

        Returns
        -------
        List of (Document, float) sorted best-first by cross-encoder score.
        """
        if not candidates:
            return candidates

        k = top_k if top_k is not None else self.top_k
        self._load()

        if self._model is None:
            # Graceful degradation: return as-is capped at top_k
            print("[Reranker] Falling back to retrieval order (no cross-encoder).")
            return candidates[:k]

        # Build (query, passage) pairs
        pairs = [(query, doc.page_content) for doc, _ in candidates]

        try:
            scores = self._model.predict(pairs)          # np.ndarray of floats
            ranked = sorted(
                zip(candidates, scores),
                key=lambda x: x[1],
                reverse=True,
            )
            print(f"[Reranker] Re-ranked {len(candidates)} → returning top {k}.")
            return [(doc_score[0], float(ce_score)) for doc_score, ce_score in ranked[:k]]
        except Exception as exc:
            print(f"[Reranker] Reranking failed ({exc}); returning retrieval order.")
            return candidates[:k]

    def rerank_provenance(
        self,
        query: str,
        provenance: List[dict],
        top_k: int | None = None,
    ) -> List[dict]:
        """
        Re-rank a list of provenance dicts (output of step8_provenance).

        Each dict must have a 'text' or 'raw_text' key.
        """
        if not provenance:
            return provenance

        k = top_k if top_k is not None else self.top_k
        self._load()

        if self._model is None:
            return provenance[:k]

        pairs = [(query, p.get("raw_text", p.get("text", ""))) for p in provenance]
        try:
            scores = self._model.predict(pairs)
            ranked = sorted(
                zip(provenance, scores),
                key=lambda x: x[1],
                reverse=True,
            )
            results = []
            for p, s in ranked[:k]:
                p = dict(p)
                p["rerank_score"] = float(s)
                results.append(p)
            print(f"[Reranker] Provenance re-ranked {len(provenance)} → top {k}.")
            return results
        except Exception as exc:
            print(f"[Reranker] Provenance reranking failed ({exc}).")
            return provenance[:k]
