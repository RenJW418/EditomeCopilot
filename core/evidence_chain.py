"""
Evidence Chain — Citation Grounding & Faithfulness Verification
===============================================================
Inspired by: MedGraphRAG (Wu et al. 2024) entity-grounded responses
             + Cancer Cell Oncology-RAG inline citation approach.

Purpose:
1. Build evidence chains linking each claim → chunk(s) → provenance
2. Inline citation with [1], [2] style markers
3. Post-hoc faithfulness verification via NLI / LLM judge
"""

from __future__ import annotations

import re
import json
from typing import Any, Dict, List, Optional, Tuple


class EvidenceChain:
    """Build and verify evidence chains for RAG responses."""

    def __init__(self, llm_client=None):
        """
        Parameters
        ----------
        llm_client : LLMClient, optional
            For faithfulness verification.  If None, only structural
            citation is performed (no LLM-based verification).
        """
        self.llm = llm_client

    # ──────────────────────────────────────────────────────────────────────
    # 1. Build Citation Map
    # ──────────────────────────────────────────────────────────────────────
    def build_citation_map(
        self, provenance_list: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict]]:
        """
        Convert provenance list into a numbered reference block and a lookup
        table that the LLM can use to insert inline citations.

        Parameters
        ----------
        provenance_list : list of dict
            Each dict should contain some of: title, authors, year, doi,
            pmid, journal, text/evidence, score.

        Returns
        -------
        reference_block : str
            Formatted reference list like a bibliography.
        citation_lookup : list of dict
            [{idx: 1, key_phrase: ..., text_snippet: ..., ref: ...}, ...]
        """
        citation_lookup: List[Dict] = []
        ref_lines: List[str] = []

        for i, prov in enumerate(provenance_list, start=1):
            # Build reference line
            authors = prov.get("authors", prov.get("author", "Unknown"))
            if isinstance(authors, list):
                authors = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
            title = prov.get("title", "Untitled")
            year = prov.get("year", prov.get("pub_year", ""))
            journal = prov.get("journal", "")
            doi = prov.get("doi", "")
            pmid = prov.get("pmid", "")

            ref_str = f"[{i}] {authors} ({year}). {title}."
            if journal:
                ref_str += f" *{journal}*."
            if doi:
                ref_str += f" doi:{doi}"
            elif pmid:
                ref_str += f" PMID:{pmid}"

            ref_lines.append(ref_str)

            # Extract key snippet for alignment
            text_snippet = prov.get("evidence", prov.get("text", prov.get("abstract", "")))
            if isinstance(text_snippet, str) and len(text_snippet) > 500:
                text_snippet = text_snippet[:500] + "..."

            citation_lookup.append({
                "idx": i,
                "ref": ref_str,
                "title": title,
                "text_snippet": text_snippet,
                "score": prov.get("score", 0),
            })

        reference_block = "\n".join(ref_lines)
        return reference_block, citation_lookup

    # ──────────────────────────────────────────────────────────────────────
    # 2. Inject Citation Instructions into LLM Prompt
    # ──────────────────────────────────────────────────────────────────────
    def format_synthesis_prompt(
        self,
        query: str,
        answer_draft: str,
        citation_lookup: List[Dict],
        reference_block: str,
        language: str = "zh",
    ) -> str:
        """
        Build a prompt that instructs the LLM to insert inline citations
        into the answer draft.

        This prompt is designed to be used as a second-pass refinement.
        """
        citation_context = "\n".join(
            f"[{c['idx']}] {c['title']} — snippet: {c['text_snippet'][:200]}"
            for c in citation_lookup
        )

        if language == "zh":
            instruction = (
                "你是一位基因编辑领域的学术写作助手。请在下面的回答文本中，"
                "为每一个关键声明（数据、实验结论、效率数字、安全性数据等）"
                "插入行内引用标注，格式为 [1], [2] 等。\n"
                "规则：\n"
                "- 只引用下面提供的参考文献\n"
                "- 如果某个声明没有对应的参考文献支持，标注为 [需要引用]\n"
                "- 不要删除或改变原文内容，只在适当位置插入引用标注\n"
                "- 在回答末尾附上完整的参考文献列表"
            )
        else:
            instruction = (
                "You are a gene-editing academic writing assistant. Insert inline "
                "citations [1], [2], etc. for every key claim (data points, "
                "experimental conclusions, efficiency numbers, safety data) in "
                "the answer below.\n"
                "Rules:\n"
                "- Only cite references provided below\n"
                "- Mark unsupported claims as [citation needed]\n"
                "- Preserve all original content; only insert citation tags\n"
                "- Append the full reference list at the end"
            )

        prompt = (
            f"{instruction}\n\n"
            f"## 用户问题 / User Query\n{query}\n\n"
            f"## 待标注回答 / Draft Answer\n{answer_draft}\n\n"
            f"## 参考文献及片段 / References & Snippets\n{citation_context}\n\n"
            f"## 完整参考文献 / Full References\n{reference_block}"
        )
        return prompt

    # ──────────────────────────────────────────────────────────────────────
    # 3. Insert Citations (LLM-based)
    # ──────────────────────────────────────────────────────────────────────
    def insert_citations(
        self,
        query: str,
        answer_draft: str,
        provenance_list: List[Dict],
        language: str = "zh",
    ) -> Tuple[str, str]:
        """
        Full pipeline: build citations → ask LLM to insert → return
        annotated answer + reference block.

        Returns
        -------
        cited_answer : str
            Answer with inline [1], [2] citations.
        reference_block : str
            Numbered reference list.
        """
        reference_block, citation_lookup = self.build_citation_map(provenance_list)

        if not self.llm or not hasattr(self.llm, "generate"):
            # Fallback: append references without inline citations
            return answer_draft + "\n\n---\n**参考文献 / References:**\n" + reference_block, reference_block

        prompt = self.format_synthesis_prompt(
            query, answer_draft, citation_lookup, reference_block, language
        )
        cited_answer = self.llm.generate(
            prompt,
            system_prompt="You are a precise academic citation assistant for gene editing.",
            timeout=60,
        )
        if cited_answer.startswith("Error"):
            return answer_draft + "\n\n---\n**References:**\n" + reference_block, reference_block

        return cited_answer, reference_block

    # ──────────────────────────────────────────────────────────────────────
    # 4. Faithfulness Verification
    # ──────────────────────────────────────────────────────────────────────
    def verify_faithfulness(
        self,
        answer: str,
        provenance_list: List[Dict],
        language: str = "zh",
    ) -> Dict[str, Any]:
        """
        Post-hoc check: does each claim in the answer have evidence support?

        Uses LLM as NLI judge: for each sentence in the answer, classify as
        SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED.

        Returns
        -------
        dict with keys:
            sentences : list of {text, verdict, supporting_ref}
            faithfulness_score : float 0-1
            unsupported_claims : list of str
        """
        if not self.llm or not hasattr(self.llm, "generate"):
            return {"sentences": [], "faithfulness_score": -1, "unsupported_claims": [],
                    "error": "No LLM available for verification"}

        # Break answer into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[。.!?！？\n])', answer) if s.strip() and len(s.strip()) > 10]

        # Prepare evidence context
        evidence_text = "\n\n".join(
            f"[{i+1}] {p.get('title', 'N/A')}: {p.get('evidence', p.get('text', p.get('abstract', '')))[:300]}"
            for i, p in enumerate(provenance_list[:10])
        )

        prompt = (
            "你是一个事实核查助手。对以下回答中的每个句子，"
            "判断它是否被提供的参考文献证据支持。\n\n"
            "对每个句子，输出一行JSON: "
            '{"sentence": "...", "verdict": "SUPPORTED|PARTIALLY_SUPPORTED|NOT_SUPPORTED", '
            '"supporting_ref": [引用编号] 或 []}\n\n'
            f"## 参考文献证据\n{evidence_text}\n\n"
            f"## 待核查句子\n" +
            "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences)) +
            "\n\n请输出JSON数组 (每个句子一个元素):"
        )

        result = self.llm.generate(prompt, timeout=60, max_tokens=4096)
        if result.startswith("Error"):
            return {"sentences": [], "faithfulness_score": -1,
                    "unsupported_claims": [], "error": result}

        # Parse result
        try:
            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                verdicts = json.loads(json_match.group())
            else:
                verdicts = []
        except json.JSONDecodeError:
            verdicts = []

        # Calculate score
        supported_count = sum(
            1 for v in verdicts
            if v.get("verdict", "").upper() in ("SUPPORTED", "PARTIALLY_SUPPORTED")
        )
        total = max(len(verdicts), 1)
        faithfulness_score = supported_count / total

        unsupported = [
            v.get("sentence", "")
            for v in verdicts
            if v.get("verdict", "").upper() == "NOT_SUPPORTED"
        ]

        return {
            "sentences": verdicts,
            "faithfulness_score": round(faithfulness_score, 3),
            "unsupported_claims": unsupported,
        }

    # ──────────────────────────────────────────────────────────────────────
    # 5. Quick Evidence Summary
    # ──────────────────────────────────────────────────────────────────────
    def summarize_evidence_chain(
        self, provenance_list: List[Dict], top_k: int = 5
    ) -> str:
        """Return a concise Markdown summary of top-K evidence sources."""
        lines = ["**Evidence Chain Summary:**"]
        for i, p in enumerate(provenance_list[:top_k], 1):
            title = p.get("title", "Untitled")
            year = p.get("year", p.get("pub_year", "N/A"))
            score = p.get("score", 0)
            evidence = p.get("evidence", p.get("text", p.get("abstract", "")))
            snippet = (evidence[:150] + "...") if len(str(evidence)) > 150 else evidence
            lines.append(f"{i}. **{title}** ({year}) [score={score:.3f}]")
            lines.append(f"   > {snippet}")
        return "\n".join(lines)
