"""
Hybrid Retrieval with MMR (Maximal Marginal Relevance)
======================================================
This module consolidates ALL retrieval logic that was previously spread across
data_pipeline.step7_retrieve and the defunct random-embedding HybridRetriever.

Key upgrades
------------
- NO random embeddings – uses the same SentenceTransformer as the pipeline.
- MMR (Maximal Marginal Relevance) for diversity-aware top-k selection.
- Reciprocal Rank Fusion (RRF) for combining dense + sparse rankings.
- Pluggable; data_pipeline can import and call directly.

Usage
-----
    retriever = HybridRetriever(embedding_model=embeddings)
    results = retriever.mmr_search(query, vector_store, bm25_index, corpus)
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Any

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity


class HybridRetriever:
    """
    Two-stage hybrid retriever: RRF fusion followed by MMR diversity selection.

    Parameters
    ----------
    embedding_model : SentenceTransformer or any object with .encode(texts) → np.ndarray
    k_rrf           : RRF constant (60 is standard)
    lambda_mmr      : trade-off between relevance (1.0) and diversity (0.0), default 0.7
    """

    def __init__(self, embedding_model=None, k_rrf: int = 60, lambda_mmr: float = 0.7):
        self.embed_model = embedding_model
        self.k_rrf = k_rrf
        self.lambda_mmr = lambda_mmr

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using the injected model."""
        if self.embed_model is None:
            raise RuntimeError("HybridRetriever: embedding_model not set.")
        if hasattr(self.embed_model, "embed_documents"):
            # LangChain HuggingFaceEmbeddings
            return np.array(self.embed_model.embed_documents(texts), dtype=np.float32)
        # raw SentenceTransformer
        return self.embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    @staticmethod
    def _rrf_fuse(
        dense_ranked: List[Tuple[Any, float]],
        sparse_ranked: List[Tuple[Any, float]],
        k_rrf: int = 60,
    ) -> List[Tuple[Any, float]]:
        """Reciprocal Rank Fusion of two ranked lists."""
        fused: dict[str, dict] = {}

        for rank, (doc, _) in enumerate(dense_ranked):
            uid = HybridRetriever._uid(doc)
            fused.setdefault(uid, {"doc": doc, "score": 0.0})
            fused[uid]["score"] += 1.0 / (k_rrf + rank + 1)

        for rank, (doc, _) in enumerate(sparse_ranked):
            uid = HybridRetriever._uid(doc)
            fused.setdefault(uid, {"doc": doc, "score": 0.0})
            fused[uid]["score"] += 1.0 / (k_rrf + rank + 1)

        ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
        return [(item["doc"], item["score"]) for item in ranked]

    @staticmethod
    def _uid(doc) -> str:
        meta = getattr(doc, "metadata", {})
        return (
            meta.get("chunk_id")
            or meta.get("checksum")
            or getattr(doc, "page_content", str(doc))[:80]
        )

    # ──────────────────────────────────────────────────────────────────────────
    # MMR
    # ──────────────────────────────────────────────────────────────────────────

    def mmr_select(
        self,
        query: str,
        candidates: List[Tuple[Any, float]],
        top_k: int = 20,
    ) -> List[Tuple[Any, float]]:
        """
        Select top_k documents from candidates using Maximal Marginal Relevance.

        MMR score = λ * sim(q, d_i) − (1−λ) * max_{d_j ∈ S} sim(d_j, d_i)

        Ensures the selected set is both relevant and diverse, preventing
        the same paper from dominating via multiple near-duplicate chunks.
        """
        if len(candidates) <= top_k:
            return candidates

        # Embed query and all candidate passages
        texts = [doc.page_content for doc, _ in candidates]
        try:
            embs = self._encode(texts)                       # (N, D)
            q_emb = self._encode([query])[0]                 # (D,)
        except Exception as exc:
            print(f"[MMR] Embedding failed ({exc}); falling back to score order.")
            return candidates[:top_k]

        # Normalise
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
        embs = embs / norms

        query_sims = embs @ q_emb                            # (N,)
        selected_indices: List[int] = []
        remaining = list(range(len(candidates)))

        for _ in range(top_k):
            if not remaining:
                break

            if not selected_indices:
                # First: pick highest query-similarity
                best = max(remaining, key=lambda i: query_sims[i])
            else:
                sel_embs = embs[selected_indices]            # (k, D)
                mmr_scores = []
                for i in remaining:
                    relevance = self.lambda_mmr * query_sims[i]
                    redundancy = (1 - self.lambda_mmr) * float(np.max(embs[i] @ sel_embs.T))
                    mmr_scores.append(relevance - redundancy)

                best = remaining[int(np.argmax(mmr_scores))]

            selected_indices.append(best)
            remaining.remove(best)

        print(f"[MMR] Selected {len(selected_indices)} diverse candidates from {len(candidates)}.")
        return [candidates[i] for i in selected_indices]

    # ──────────────────────────────────────────────────────────────────────────
    # Main public API
    # ──────────────────────────────────────────────────────────────────────────

    def hybrid_search(
        self,
        query: str,
        vector_store,
        bm25_index: Optional[BM25Okapi],
        corpus: List[Any],
        top_k: int = 20,
        fetch_k: int = 100,
        use_mmr: bool = True,
    ) -> List[Tuple[Any, float]]:
        """
        Full hybrid retrieval pipeline:
          1. Dense FAISS top-fetch_k
          2. BM25 top-fetch_k
          3. RRF fusion
          4. MMR diversity selection → top_k

        Parameters
        ----------
        vector_store : FAISS LangChain store
        bm25_index   : BM25Okapi built over corpus
        corpus       : list of LangChain Documents (same order as BM25 index)
        top_k        : final number to return after MMR
        fetch_k      : number to retrieve per source before fusion
        use_mmr      : whether to apply MMR (True in production)
        """
        # ── Dense ──────────────────────────────────────────────────────────────
        dense_results: List[Tuple[Any, float]] = []
        if vector_store:
            try:
                cap = min(fetch_k, max(1, len(corpus))) if corpus else fetch_k
                dense_results = vector_store.similarity_search_with_score(query, k=cap)
            except Exception as exc:
                print(f"[Hybrid] Dense search failed: {exc}")

        # ── Sparse (BM25) ──────────────────────────────────────────────────────
        sparse_results: List[Tuple[Any, float]] = []
        if bm25_index and corpus:
            tokens = query.lower().split()
            scores = bm25_index.get_scores(tokens)
            top_idx = np.argsort(scores)[::-1][:fetch_k]
            for idx in top_idx:
                if scores[idx] > 0:
                    sparse_results.append((corpus[idx], float(scores[idx])))

        # ── Fusion ─────────────────────────────────────────────────────────────
        fused = self._rrf_fuse(dense_results, sparse_results, k_rrf=self.k_rrf)

        if not fused:
            return []

        # ── MMR ────────────────────────────────────────────────────────────────
        if use_mmr and self.embed_model is not None:
            return self.mmr_select(query, fused, top_k=top_k)
        return fused[:top_k]


if __name__ == "__main__":
    # Quick test with the real HybridRetriever
    print("HybridRetriever module loaded successfully.")
    print("Use HybridRetriever(embedding_model=...) with .hybrid_search() for production retrieval.")
