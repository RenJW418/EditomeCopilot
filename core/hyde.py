"""
HyDE – Hypothetical Document Embeddings
========================================
Reference: Gao et al. 2022 "Precise Zero-Shot Dense Retrieval without Relevance Labels"
           https://arxiv.org/abs/2212.10496

Intuition
---------
Sparse / bi-encoder models struggle when the user query is short and uses
lay / Chinese vocabulary, while the corpus is full of technical English.

HyDE solves this by asking the LLM to *hallucinate* a hypothetical answer,
then using the embedding of that answer (which shares vocabulary with the
corpus) as the query vector.

Pipeline
--------
  user_query → LLM → hypothetical_answer → embed → FAISS search
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm_client import LLMClient


# Biomedical HyDE prompt template
_HYDE_PROMPT = """\
You are an expert molecular biologist and gene-editing specialist.
A researcher asks: "{query}"

Write a dense, information-rich scientific paragraph (≈150 words) that DIRECTLY
answers this question, as if it were an excerpt from a high-quality research
paper or review (Nature, Science, Cell, NEJM level).

Use precise technical terminology: gene names, editing efficiency numbers,
delivery modalities, cell types, safety metrics, etc.
Do NOT mention that this is hypothetical. Write as a factual statement.
Output ONLY the paragraph – no headers, no preamble.
"""


class HypotheticalDocumentEmbedder:
    """
    Generates a hypothetical answer and returns it for use as a search query
    vector, significantly improving recall for technical biomedical queries.
    """

    def __init__(self, llm_client: "LLMClient", enabled: bool = True):
        self.llm = llm_client
        self.enabled = enabled

    # -------------------------------------------------------------------------
    def generate(self, query: str) -> str:
        """
        Return the hypothetical document text.
        Falls back to the original query if LLM is unavailable.
        """
        if not self.enabled or not self.llm.client:
            return query

        prompt = _HYDE_PROMPT.format(query=query)
        try:
            hyp = self.llm.generate(
                prompt,
                system_prompt="You are a scientific literature expert. Respond only with the paragraph.",
                enable_thinking=False,
                timeout=20,
                max_tokens=250,
            )
            if hyp and not str(hyp).startswith("Error"):
                # Sanitise: remove any stray markdown artifacts
                hyp = re.sub(r"```[a-z]*", "", hyp).strip()
                print(f"[HyDE] Generated hypothetical document ({len(hyp)} chars).")
                return hyp
        except Exception as exc:
            print(f"[HyDE] LLM call failed ({exc}); using original query.")

        return query

    # -------------------------------------------------------------------------
    def enhanced_query(self, query: str) -> str:
        """
        Concatenate original query + hypothetical answer.
        This gives the bi-encoder both the user intent and the assumed answer
        vocabulary, empirically improving recall vs. using either alone.
        """
        hyp = self.generate(query)
        if hyp == query:
            return query
        return f"{query}\n\n{hyp}"
