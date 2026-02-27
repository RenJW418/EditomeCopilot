"""
Baseline Systems for Comparative Evaluation
=============================================
Four baselines covering the spectrum from no-retrieval to standard RAG:

  B1  LLM-Only       – Qwen-max direct generation, no retrieval
  B2  Naive RAG      – FAISS top-k + LLM synthesis (no domain modules)
  B3  PubMed RAG     – PubMed API real-time search + LLM synthesis
  B4  LangChain RAG  – Standard LangChain RetrievalQA chain

Each baseline implements a unified interface:
  baseline.answer(question: str) -> dict {answer, contexts, latency}
"""

from __future__ import annotations

import os
import re
import time
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaselineSystem(ABC):
    """Abstract base for all baseline systems."""

    name: str = "BaselineSystem"
    description: str = ""

    @abstractmethod
    def answer(self, question: str) -> Dict[str, Any]:
        """
        Generate answer for a question.

        Returns dict with keys:
          - answer: str
          - contexts: List[str]  (retrieved chunks, empty if none)
          - latency: float  (seconds)
        """
        ...

    def batch_answer(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Answer multiple questions sequentially."""
        return [self.answer(q) for q in questions]


# ---------------------------------------------------------------------------
# B1: LLM-Only (No Retrieval)
# ---------------------------------------------------------------------------

_LLM_ONLY_SYSTEM = (
    "You are an expert in gene editing and genomic medicine. "
    "Answer the question comprehensively based on your training knowledge. "
    "Use Markdown formatting with headers and bullet points. "
    "Always respond in the same language as the question."
)


class LLMOnlyBaseline(BaselineSystem):
    """Direct LLM generation without any retrieval or domain tools."""

    name = "LLM-Only"
    description = "Direct Qwen-max generation without retrieval augmentation"

    def __init__(self, llm_client: Any = None):
        if llm_client is None:
            from core.llm_client import LLMClient
            llm_client = LLMClient()
        self.llm = llm_client

    def answer(self, question: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            raw = self.llm.generate(
                question,
                system_prompt=_LLM_ONLY_SYSTEM,
                max_tokens=2000,
                timeout=120,
            )
        except Exception as e:
            raw = f"[Error] {e}"
        latency = time.perf_counter() - t0
        return {"answer": raw, "contexts": [], "latency": latency}


# ---------------------------------------------------------------------------
# B2: Naive RAG (FAISS + LLM, no domain modules)
# ---------------------------------------------------------------------------

_NAIVE_RAG_SYSTEM = (
    "You are a scientific assistant. Answer the question based ONLY on the "
    "provided context. If insufficient, say so. "
    "Respond in the same language as the question."
)

_NAIVE_RAG_USER = """Context:
{context}

Question: {question}

Provide a detailed answer based on the context above:"""


class NaiveRAGBaseline(BaselineSystem):
    """Standard FAISS vector retrieval + LLM synthesis, no domain modules."""

    name = "Naive-RAG"
    description = "FAISS top-k retrieval + LLM synthesis without domain-specific modules"

    def __init__(
        self,
        llm_client: Any = None,
        data_pipeline: Any = None,
        top_k: int = 10,
    ):
        if llm_client is None:
            from core.llm_client import LLMClient
            llm_client = LLMClient()
        self.llm = llm_client
        self.top_k = top_k

        if data_pipeline is None:
            try:
                from core.data_pipeline import GeneEditingDataPipeline
                data_pipeline = GeneEditingDataPipeline(base_dir="data")
            except Exception as e:
                print(f"[NaiveRAG] Data pipeline init failed: {e}")
                data_pipeline = None
        self.pipeline = data_pipeline

    def _retrieve(self, question: str) -> List[str]:
        """Simple FAISS similarity search."""
        if not self.pipeline:
            return []
        try:
            results = self.pipeline.step7_retrieve(question, top_k=self.top_k)
            chunks = []
            for item in results:
                if isinstance(item, tuple) and len(item) >= 1:
                    doc = item[0]
                    text = doc.page_content if hasattr(doc, "page_content") else str(doc)
                elif isinstance(item, dict):
                    text = item.get("text", item.get("evidence", str(item)))
                else:
                    text = str(item)
                chunks.append(text[:1000])
            return chunks
        except Exception as e:
            print(f"[NaiveRAG] Retrieval error: {e}")
            return []

    def answer(self, question: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        contexts = self._retrieve(question)

        context_str = "\n---\n".join(contexts[:self.top_k]) if contexts else "No relevant context found."
        prompt = _NAIVE_RAG_USER.format(context=context_str[:4000], question=question)

        try:
            raw = self.llm.generate(
                prompt,
                system_prompt=_NAIVE_RAG_SYSTEM,
                max_tokens=2000,
                timeout=120,
            )
        except Exception as e:
            raw = f"[Error] {e}"

        latency = time.perf_counter() - t0
        return {"answer": raw, "contexts": contexts, "latency": latency}


# ---------------------------------------------------------------------------
# B3: PubMed RAG (Real-time PubMed search + LLM)
# ---------------------------------------------------------------------------

_PUBMED_SYSTEM = (
    "You are a biomedical research assistant. Synthesize the PubMed search "
    "results into a comprehensive answer. Cite PMIDs when possible. "
    "Respond in the same language as the question."
)

_PUBMED_USER = """PubMed Search Results:
{results}

Question: {question}

Synthesize a comprehensive answer from the search results:"""


class PubMedRAGBaseline(BaselineSystem):
    """Real-time PubMed API search + LLM synthesis."""

    name = "PubMed-RAG"
    description = "Real-time PubMed E-utils search + LLM synthesis"

    def __init__(self, llm_client: Any = None, max_results: int = 15):
        if llm_client is None:
            from core.llm_client import LLMClient
            llm_client = LLMClient()
        self.llm = llm_client
        self.max_results = max_results

    def _translate_to_english(self, question: str) -> str:
        """Translate Chinese question to English for PubMed search."""
        if re.search(r'[\u4e00-\u9fff]', question):
            try:
                result = self.llm.generate(
                    f"Translate this gene editing question to English for PubMed search. "
                    f"Return ONLY the English query, no explanation:\n{question}",
                    max_tokens=200,
                    timeout=20,
                )
                return result.strip()
            except Exception:
                return question
        return question

    def _search_pubmed(self, query: str) -> List[Dict]:
        """Search PubMed E-Utils API."""
        import urllib.request
        import urllib.parse
        import xml.etree.ElementTree as ET

        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        email = os.getenv("NCBI_EMAIL", "researcher@example.com")

        try:
            # Step 1: Search
            search_url = (
                f"{base}/esearch.fcgi?"
                f"db=pubmed&term={urllib.parse.quote(query)}"
                f"&retmax={self.max_results}&sort=relevance&email={email}"
            )
            with urllib.request.urlopen(search_url, timeout=15) as resp:
                search_xml = resp.read().decode()

            root = ET.fromstring(search_xml)
            pmids = [id_elem.text for id_elem in root.findall(".//Id")][:self.max_results]
            if not pmids:
                return []

            # Step 2: Fetch abstracts
            ids_str = ",".join(pmids)
            fetch_url = (
                f"{base}/efetch.fcgi?"
                f"db=pubmed&id={ids_str}&rettype=abstract&retmode=xml&email={email}"
            )
            with urllib.request.urlopen(fetch_url, timeout=20) as resp:
                fetch_xml = resp.read().decode()

            articles = []
            art_root = ET.fromstring(fetch_xml)
            for article in art_root.findall(".//PubmedArticle"):
                title_el = article.find(".//ArticleTitle")
                abstract_el = article.find(".//AbstractText")
                pmid_el = article.find(".//PMID")
                title = title_el.text if title_el is not None and title_el.text else ""
                abstract = abstract_el.text if abstract_el is not None and abstract_el.text else ""
                pmid = pmid_el.text if pmid_el is not None and pmid_el.text else ""
                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract[:500],
                })
            return articles
        except Exception as e:
            print(f"[PubMedRAG] Search error: {e}")
            return []

    def answer(self, question: str) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # Translate if Chinese
        en_query = self._translate_to_english(question)

        # Search PubMed
        articles = self._search_pubmed(en_query)

        # Format results
        contexts = []
        result_text = ""
        for i, art in enumerate(articles, 1):
            snippet = f"[{i}] PMID:{art['pmid']} - {art['title']}\n{art['abstract']}"
            result_text += snippet + "\n\n"
            contexts.append(snippet)

        if not result_text:
            result_text = "No PubMed results found."

        # LLM synthesis
        prompt = _PUBMED_USER.format(results=result_text[:5000], question=question)
        try:
            raw = self.llm.generate(
                prompt,
                system_prompt=_PUBMED_SYSTEM,
                max_tokens=2000,
                timeout=120,
            )
        except Exception as e:
            raw = f"[Error] {e}"

        latency = time.perf_counter() - t0
        return {"answer": raw, "contexts": contexts, "latency": latency}


# ---------------------------------------------------------------------------
# B4: LangChain-Style RAG
# ---------------------------------------------------------------------------

_LANGCHAIN_SYSTEM = (
    "You are a helpful assistant. Use the following pieces of context to answer "
    "the question. If you don't know the answer, say so. "
    "Respond in the same language as the question."
)

_LANGCHAIN_USER = """Context:
{context}

Question: {question}

Helpful Answer:"""


class LangChainRAGBaseline(BaselineSystem):
    """
    Simulates a standard LangChain RetrievalQA pipeline:
    embedding search → stuff prompt → LLM.
    Uses the same embedding + FAISS as the main system but no reranking,
    HyDE, knowledge graph, or domain modules.
    """

    name = "LangChain-RAG"
    description = "Standard embedding retrieval + stuff prompt (no reranking or domain modules)"

    def __init__(
        self,
        llm_client: Any = None,
        data_pipeline: Any = None,
        top_k: int = 5,
    ):
        if llm_client is None:
            from core.llm_client import LLMClient
            llm_client = LLMClient()
        self.llm = llm_client
        self.top_k = top_k

        if data_pipeline is None:
            try:
                from core.data_pipeline import GeneEditingDataPipeline
                data_pipeline = GeneEditingDataPipeline(base_dir="data")
            except Exception as e:
                print(f"[LangChainRAG] Data pipeline init failed: {e}")
                data_pipeline = None
        self.pipeline = data_pipeline

    def answer(self, question: str) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # Simple retrieval (no HyDE, no reranking)
        contexts = []
        if self.pipeline:
            try:
                results = self.pipeline.step7_retrieve(question, top_k=self.top_k)
                for item in results:
                    if isinstance(item, tuple) and len(item) >= 1:
                        doc = item[0]
                        text = doc.page_content if hasattr(doc, "page_content") else str(doc)
                    elif isinstance(item, dict):
                        text = item.get("text", str(item))
                    else:
                        text = str(item)
                    contexts.append(text[:800])
            except Exception as e:
                print(f"[LangChainRAG] Retrieval error: {e}")

        # Stuff prompt (LangChain default)
        context_str = "\n\n".join(contexts) if contexts else "No context available."
        prompt = _LANGCHAIN_USER.format(
            context=context_str[:3000], question=question
        )

        try:
            raw = self.llm.generate(
                prompt,
                system_prompt=_LANGCHAIN_SYSTEM,
                max_tokens=1500,
                timeout=120,
            )
        except Exception as e:
            raw = f"[Error] {e}"

        latency = time.perf_counter() - t0
        return {"answer": raw, "contexts": contexts, "latency": latency}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_all_baselines(
    llm_client: Any = None,
    data_pipeline: Any = None,
) -> Dict[str, BaselineSystem]:
    """Create all 4 baseline systems.

    Returns dict: name -> BaselineSystem
    """
    if llm_client is None:
        from core.llm_client import LLMClient
        llm_client = LLMClient()

    return {
        "LLM-Only": LLMOnlyBaseline(llm_client=llm_client),
        "Naive-RAG": NaiveRAGBaseline(llm_client=llm_client, data_pipeline=data_pipeline),
        "PubMed-RAG": PubMedRAGBaseline(llm_client=llm_client),
        "LangChain-RAG": LangChainRAGBaseline(llm_client=llm_client, data_pipeline=data_pipeline),
    }
