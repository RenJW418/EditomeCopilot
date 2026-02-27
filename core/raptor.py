"""
RAPTOR – Recursive Abstractive Processing for Tree-Organised Retrieval
=======================================================================
Reference: Sarthi et al. 2024  "RAPTOR: Recursive Abstractive Processing
           for Tree-Organised Retrieval"  (arXiv 2401.18059)

Architecture
------------
  Leaf nodes  : original text chunks (already in FAISS)
  Branch nodes: LLM summaries of semantically-similar clusters
  Root node   : global summary

At query time we retrieve from ALL tree levels simultaneously, giving the
LLM both detailed evidence AND high-level synthesis.

This is especially powerful for "综述" / "overview" queries where the user
needs cross-paper insight within 128k-token context windows.

Implementation notes
--------------------
- Clustering uses UMAP (dimensionality reduction) + Gaussian Mixture Model
  (soft-assignment clustering).  Falls back to k-means if UMAP is absent.
- Summaries are cached on disk so subsequent runs are cheap.
- Tree depth is configurable; default depth=2 (leaf → branch → root).
"""

from __future__ import annotations

import json
import os
import hashlib
import pickle
from typing import List, Dict, Any, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Cluster helpers
# ---------------------------------------------------------------------------

def _cluster_embeddings(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Returns integer cluster labels for each document embedding.
    Tries UMAP + GMM first; falls back to k-means.
    """
    try:
        import umap  # type: ignore
        from sklearn.mixture import GaussianMixture

        reducer = umap.UMAP(n_components=min(10, embeddings.shape[1] - 1), random_state=42)
        reduced = reducer.fit_transform(embeddings)

        gm = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = gm.fit_predict(reduced)
        return labels

    except Exception as exc:
        print(f"[RAPTOR] UMAP/GMM unavailable ({exc}); using k-means.")
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return km.fit_predict(embeddings)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RaptorIndexer:
    """
    Builds a RAPTOR summary tree on top of existing chunk embeddings.

    Parameters
    ----------
    llm_client      : LLMClient  – used for generating cluster summaries
    embedding_fn    : callable   – takes a list[str] → np.ndarray (shape N×D)
    cache_dir       : path to persist the tree between runs
    max_clusters    : target cluster count per tree level
    max_depth       : how many levels to build (typically 2)
    """

    CACHE_FILE = "raptor_tree.pkl"

    def __init__(
        self,
        llm_client: "LLMClient",
        embedding_fn,
        cache_dir: str = "data/raptor_cache",
        max_clusters: int = 10,
        max_depth: int = 2,
    ):
        self.llm = llm_client
        self.embed = embedding_fn
        self.cache_dir = cache_dir
        self.max_clusters = max_clusters
        self.max_depth = max_depth
        os.makedirs(cache_dir, exist_ok=True)

        self.tree: List[Dict[str, Any]] = []   # list of {"text", "level", "cluster"}
        self._load_cache()

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    def _cache_path(self) -> str:
        return os.path.join(self.cache_dir, self.CACHE_FILE)

    def _load_cache(self):
        p = self._cache_path()
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    self.tree = pickle.load(f)
                print(f"[RAPTOR] Loaded tree with {len(self.tree)} summary nodes.")
            except Exception:
                self.tree = []

    def _save_cache(self):
        with open(self._cache_path(), "wb") as f:
            pickle.dump(self.tree, f)

    # -------------------------------------------------------------------------
    # Build tree
    # -------------------------------------------------------------------------
    def build(self, texts: List[str], force_rebuild: bool = False):
        """
        Build the RAPTOR tree from a flat list of chunk texts.
        Skips if tree already cached unless force_rebuild=True.
        """
        if self.tree and not force_rebuild:
            print("[RAPTOR] Tree already built. Use force_rebuild=True to regenerate.")
            return

        if not texts:
            print("[RAPTOR] No texts supplied; skipping build.")
            return

        print(f"[RAPTOR] Building tree from {len(texts)} leaf chunks (depth={self.max_depth})…")

        current_texts = texts
        all_nodes: List[Dict[str, Any]] = []

        for level in range(1, self.max_depth + 1):
            if len(current_texts) < 3:
                break

            n_clusters = min(self.max_clusters, max(2, len(current_texts) // 5))
            print(f"[RAPTOR] Level {level}: clustering {len(current_texts)} texts into {n_clusters} clusters…")

            try:
                embeddings = self.embed(current_texts)
                labels = _cluster_embeddings(embeddings, n_clusters)
            except Exception as exc:
                print(f"[RAPTOR] Clustering failed at level {level}: {exc}")
                break

            cluster_summaries: List[str] = []
            for cid in range(n_clusters):
                cluster_texts = [t for t, l in zip(current_texts, labels) if l == cid]
                if not cluster_texts:
                    continue
                summary = self._summarise(cluster_texts, level=level, cluster_id=cid)
                if summary:
                    node = {
                        "text": summary,
                        "level": level,
                        "cluster": cid,
                        "n_children": len(cluster_texts),
                        "checksum": hashlib.md5(summary.encode()).hexdigest(),
                    }
                    all_nodes.append(node)
                    cluster_summaries.append(summary)

            current_texts = cluster_summaries  # next level clusters the summaries

        self.tree = all_nodes
        self._save_cache()
        print(f"[RAPTOR] Tree built with {len(self.tree)} summary nodes.")

    # -------------------------------------------------------------------------
    # Summarise a cluster
    # -------------------------------------------------------------------------
    def _summarise(self, texts: List[str], level: int, cluster_id: int) -> Optional[str]:
        if not self.llm.client:
            # Fallback: simple concatenation (no LLM)
            return " ".join(t[:200] for t in texts[:3]) + " [RAPTOR-no-LLM]"

        # Keep within typical context limit
        combined = "\n\n---\n\n".join(t[:800] for t in texts[:15])
        prompt = f"""You are a senior scientific editor specialising in gene editing.
Synthesise the following {len(texts)} research excerpts (Level {level} cluster {cluster_id}) into a
dense, information-rich summary paragraph (≈200 words). Preserve key numbers,
gene names, technology names, and clinical findings. Write in flowing prose.

Excerpts:
{combined}

Summary paragraph:"""

        try:
            result = self.llm.generate(
                prompt,
                system_prompt="You are an expert scientific summariser.",
                enable_thinking=False,
                timeout=60,
                max_tokens=350,
            )
            if result and not str(result).startswith("Error"):
                return result.strip()
        except Exception as exc:
            print(f"[RAPTOR] Summary generation failed: {exc}")
        return None

    # -------------------------------------------------------------------------
    # Retrieve from tree
    # -------------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant summary nodes from the tree.
        Uses cosine similarity against the query embedding.
        """
        if not self.tree:
            return []

        try:
            q_emb = self.embed([query])[0]            # shape (D,)
            tree_texts = [n["text"] for n in self.tree]
            tree_embs = self.embed(tree_texts)         # shape (N, D)

            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity([q_emb], tree_embs)[0]

            top_idx = np.argsort(sims)[::-1][:top_k]
            results = []
            for i in top_idx:
                node = dict(self.tree[i])
                node["score"] = float(sims[i])
                results.append(node)

            print(f"[RAPTOR] Retrieved {len(results)} summary nodes.")
            return results
        except Exception as exc:
            print(f"[RAPTOR] Retrieval failed: {exc}")
            return []

    def to_provenance(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert RAPTOR nodes to the provenance dict format used by AgenticRAG."""
        results = []
        for n in nodes:
            results.append({
                "evidence": f"[RAPTOR-L{n['level']}] Cluster summary ({n.get('n_children', '?')} papers)",
                "text": n["text"],
                "raw_text": n["text"],
                "score": n.get("score", 0.0),
                "doi": None,
                "structured_data": {},
                "is_raptor": True,
            })
        return results
