"""
Semantic Chunker 2.0 — Embedding-Based Adaptive Chunking
==========================================================
Purpose: Replace naive fixed-size splitting with embedding-based semantic
boundary detection for user-uploaded full-text documents.

Strategy:
1. Split text into sentences
2. Compute sentence embeddings
3. Detect semantic breakpoints via cosine similarity drop
4. Merge adjacent sentences into coherent semantic chunks
5. Respect max chunk size constraints

Note: This is for user-uploaded full-text docs (PDFs, etc.).
The primary knowledge base (86K+ papers) uses one-article-per-chunk
strategy (see process_knowledge_base.py).
"""

from __future__ import annotations

import re
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class SemanticChunker:
    """Embedding-based semantic chunking for long documents."""

    def __init__(
        self,
        model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        similarity_threshold: float = 0.5,
        max_chunk_chars: int = 3000,
        min_chunk_chars: int = 200,
        window_size: int = 3,
    ):
        """
        Parameters
        ----------
        model_name : str
            Sentence-transformer model for embeddings.
        similarity_threshold : float
            Cosine similarity below this triggers a chunk boundary.
        max_chunk_chars : int
            Hard maximum for any single chunk.
        min_chunk_chars : int
            Minimum chars — short chunks get merged with neighbors.
        window_size : int
            Number of sentences to average for smoothed similarity.
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.window_size = window_size
        self._model = None

    def _get_model(self):
        """Lazy-load embedding model."""
        if self._model is None:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers required: pip install sentence-transformers")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ─── Sentence Splitting ───────────────────────────────────────────────

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """
        Split text into sentences. Handles both English and Chinese.
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Split on sentence boundaries
        # English: .!? followed by space or end + Chinese: 。！？
        sentences = re.split(
            r'(?<=[.!?。！？])\s+|(?<=[.!?。！？])(?=[A-Z\u4e00-\u9fff])',
            text,
        )

        # Filter out very short fragments
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return sentences

    # ─── Similarity Computation ───────────────────────────────────────────

    def compute_similarities(self, sentences: List[str]) -> np.ndarray:
        """
        Compute cosine similarity between consecutive sentence groups.

        Returns array of shape (N-1,) where each element is the
        similarity between sentences[i] and sentences[i+1].
        """
        if len(sentences) < 2:
            return np.array([])

        model = self._get_model()
        embeddings = model.encode(sentences, show_progress_bar=False, batch_size=32)
        embeddings = np.array(embeddings)

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

        # Windowed similarity (average over window_size)
        n = len(sentences)
        similarities = np.zeros(n - 1)
        for i in range(n - 1):
            # Average embedding for left window
            left_start = max(0, i - self.window_size + 1)
            left_emb = embeddings[left_start:i + 1].mean(axis=0)

            # Average embedding for right window
            right_end = min(n, i + 1 + self.window_size)
            right_emb = embeddings[i + 1:right_end].mean(axis=0)

            # Cosine similarity
            similarities[i] = np.dot(left_emb, right_emb)

        return similarities

    # ─── Breakpoint Detection ─────────────────────────────────────────────

    def detect_breakpoints(
        self, similarities: np.ndarray, method: str = "threshold"
    ) -> List[int]:
        """
        Detect semantic breakpoints from similarity array.

        Parameters
        ----------
        similarities : np.ndarray
            Consecutive sentence similarities.
        method : str
            'threshold' — break where similarity < threshold
            'gradient' — break at local minima with gradient detection

        Returns
        -------
        list of int — indices where breaks should occur.
        """
        if len(similarities) == 0:
            return []

        if method == "threshold":
            breaks = [i for i, s in enumerate(similarities) if s < self.similarity_threshold]
        elif method == "gradient":
            # Find local minima (significant drops)
            mean_sim = similarities.mean()
            std_sim = similarities.std() if len(similarities) > 1 else 0.1
            threshold = mean_sim - std_sim
            breaks = []
            for i in range(1, len(similarities) - 1):
                if (similarities[i] < similarities[i-1] and
                    similarities[i] < similarities[i+1] and
                    similarities[i] < threshold):
                    breaks.append(i)
        else:
            breaks = [i for i, s in enumerate(similarities) if s < self.similarity_threshold]

        return breaks

    # ─── Main Chunking Pipeline ───────────────────────────────────────────

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        method: str = "threshold",
    ) -> List[Dict[str, Any]]:
        """
        Main entry point: split text into semantic chunks.

        Parameters
        ----------
        text : str
            Full document text.
        metadata : dict, optional
            Base metadata to attach to each chunk.
        method : str
            Breakpoint detection method: 'threshold' or 'gradient'.

        Returns
        -------
        list of dict, each with 'content', 'metadata', 'chunk_idx'.
        """
        if not text or len(text.strip()) < self.min_chunk_chars:
            return [{
                "content": text.strip(),
                "metadata": {**(metadata or {}), "chunk_idx": 0, "chunk_method": "single"},
                "chunk_idx": 0,
            }]

        # Step 1: Split into sentences
        sentences = self.split_sentences(text)
        if len(sentences) <= 2:
            return [{
                "content": text.strip(),
                "metadata": {**(metadata or {}), "chunk_idx": 0, "chunk_method": "too_short"},
                "chunk_idx": 0,
            }]

        # Step 2: Compute similarities
        similarities = self.compute_similarities(sentences)

        # Step 3: Detect breakpoints
        breaks = self.detect_breakpoints(similarities, method=method)

        # Step 4: Group sentences into chunks
        chunks_raw = []
        prev = 0
        for bp in breaks:
            chunk_sentences = sentences[prev:bp + 1]
            if chunk_sentences:
                chunks_raw.append(" ".join(chunk_sentences))
            prev = bp + 1
        # Last chunk
        if prev < len(sentences):
            chunks_raw.append(" ".join(sentences[prev:]))

        # Step 5: Enforce size constraints
        final_chunks = self._enforce_size_constraints(chunks_raw)

        # Step 6: Package with metadata
        results = []
        for i, chunk_text in enumerate(final_chunks):
            chunk_meta = {
                **(metadata or {}),
                "chunk_idx": i,
                "chunk_total": len(final_chunks),
                "chunk_chars": len(chunk_text),
                "chunk_method": "semantic",
            }
            results.append({
                "content": chunk_text,
                "metadata": chunk_meta,
                "chunk_idx": i,
            })

        return results

    def _enforce_size_constraints(self, chunks: List[str]) -> List[str]:
        """
        Merge too-small chunks and split too-large chunks.
        """
        # Pass 1: Merge small chunks with next
        merged = []
        buffer = ""
        for chunk in chunks:
            if buffer:
                combined = buffer + " " + chunk
                if len(combined) <= self.max_chunk_chars:
                    buffer = combined
                else:
                    merged.append(buffer)
                    buffer = chunk
            else:
                buffer = chunk

            if len(buffer) >= self.min_chunk_chars:
                merged.append(buffer)
                buffer = ""

        if buffer:
            if merged:
                # Merge with last chunk if possible
                combined = merged[-1] + " " + buffer
                if len(combined) <= self.max_chunk_chars:
                    merged[-1] = combined
                else:
                    merged.append(buffer)
            else:
                merged.append(buffer)

        # Pass 2: Split oversized chunks
        final = []
        for chunk in merged:
            if len(chunk) <= self.max_chunk_chars:
                final.append(chunk)
            else:
                # Split at sentence boundaries within chunk
                sub_sentences = self.split_sentences(chunk)
                sub_buffer = ""
                for sent in sub_sentences:
                    if sub_buffer and len(sub_buffer) + len(sent) + 1 > self.max_chunk_chars:
                        final.append(sub_buffer)
                        sub_buffer = sent
                    else:
                        sub_buffer = (sub_buffer + " " + sent).strip() if sub_buffer else sent
                if sub_buffer:
                    final.append(sub_buffer)

        return final

    # ─── Batch Processing ─────────────────────────────────────────────────

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "text",
        method: str = "threshold",
    ) -> List[Dict[str, Any]]:
        """
        Chunk a batch of documents.

        Parameters
        ----------
        documents : list of dict
            Each should have a text field and optional metadata.
        text_field : str
            Key to read full text from.

        Returns
        -------
        list of chunk dicts.
        """
        all_chunks = []
        for doc in documents:
            text = doc.get(text_field, "")
            meta = {k: v for k, v in doc.items() if k != text_field}
            chunks = self.chunk(text, metadata=meta, method=method)
            all_chunks.extend(chunks)
        return all_chunks

    # ─── Statistics ───────────────────────────────────────────────────────

    def analyze_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return statistics about chunk size distribution."""
        sizes = [len(c.get("content", "")) for c in chunks]
        if not sizes:
            return {"count": 0}

        arr = np.array(sizes)
        return {
            "count": len(sizes),
            "mean_chars": round(float(arr.mean()), 1),
            "median_chars": round(float(np.median(arr)), 1),
            "p5_chars": round(float(np.percentile(arr, 5)), 1),
            "p95_chars": round(float(np.percentile(arr, 95)), 1),
            "min_chars": int(arr.min()),
            "max_chars": int(arr.max()),
        }
