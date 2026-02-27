"""
Query Decomposer
================
Breaks a complex multi-faceted question into atomic sub-queries, each of
which is independently retrievable. Results are retrieved in parallel and
merged via Reciprocal Rank Fusion.

Why this matters
----------------
"Compare prime editing vs base editing efficiency, safety, and clinical
translation in haematological diseases" = 5 independent sub-queries.
A monolithic retrieval often misses the tail sub-topics.

Reference
---------
Inspired by "Self-RAG" and "FLARE" papers, adapted for biomedical domain.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from core.llm_client import LLMClient


_DECOMPOSE_PROMPT = """\
You are an expert biomedical query analyst.

Break the following complex gene-editing / precision-medicine question into
2-5 ATOMIC sub-queries. Each sub-query must be independently answerable from
a scientific literature database.

Rules:
- Output ONLY a JSON array of strings. No preamble.
- Each sub-query: ≤ 20 words, focused on ONE concept.
- Preserve specifics (gene names, mutations, cell types, technologies).
- If the question is already atomic (≤ 2 concepts), return a 1-element array.

Original question: "{query}"

JSON array:"""


class QueryDecomposer:
    """Decomposes a complex query into retrievable sub-queries."""

    def __init__(self, llm_client: "LLMClient", enabled: bool = True):
        self.llm = llm_client
        self.enabled = enabled

    # -------------------------------------------------------------------------
    def decompose(self, query: str) -> List[str]:
        """
        Return a list of sub-queries.
        Always includes the original query as the first element (fallback).
        """
        if not self.enabled or not self.llm.client:
            return [query]

        prompt = _DECOMPOSE_PROMPT.format(query=query)
        try:
            raw = self.llm.generate(
                prompt,
                system_prompt="You are a query analysis agent. Output only JSON.",
                enable_thinking=False,
                timeout=15,
                max_tokens=300,
            )
            if not raw or str(raw).startswith("Error"):
                return [query]

            sub_queries = self._parse(raw)
            if sub_queries:
                # Always ensure the original question is in the list
                if query not in sub_queries:
                    sub_queries.insert(0, query)
                print(f"[QueryDecomposer] Decomposed into {len(sub_queries)} sub-queries.")
                return sub_queries
        except Exception as exc:
            print(f"[QueryDecomposer] Failed ({exc}).")

        return [query]

    # -------------------------------------------------------------------------
    @staticmethod
    def _parse(text: str) -> List[str]:
        cleaned = re.sub(r"```[a-z]*", "", str(text)).strip()

        # Direct
        try:
            result = json.loads(cleaned)
            if isinstance(result, list):
                return [s for s in result if isinstance(s, str) and s.strip()]
        except Exception:
            pass

        # Extract array
        m = re.search(r"\[[\s\S]*?\]", cleaned)
        if m:
            try:
                result = json.loads(m.group(0))
                if isinstance(result, list):
                    return [s for s in result if isinstance(s, str) and s.strip()]
            except Exception:
                pass

        return []

    # -------------------------------------------------------------------------
    @staticmethod
    def fuse_results(results_per_subquery: List[List[tuple]], k_rrf: int = 60) -> List[tuple]:
        """
        Reciprocal Rank Fusion across results from multiple sub-queries.

        Parameters
        ----------
        results_per_subquery : List[ List[(Document, score)] ]
        k_rrf : RRF constant (60 is standard)

        Returns
        -------
        Merged, deduplicated list of (Document, rrf_score) sorted best-first.
        """
        fused: dict[str, dict] = {}

        for results in results_per_subquery:
            for rank, (doc, _score) in enumerate(results):
                uid = (
                    doc.metadata.get("chunk_id")
                    or doc.metadata.get("checksum")
                    or doc.page_content[:80]
                )
                if uid not in fused:
                    fused[uid] = {"doc": doc, "score": 0.0}
                fused[uid]["score"] += 1.0 / (k_rrf + rank + 1)

        sorted_docs = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
        return [(item["doc"], item["score"]) for item in sorted_docs]
