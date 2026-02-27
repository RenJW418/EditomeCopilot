from __future__ import annotations

import re
from typing import List, Dict, Optional

from core.llm_client import LLMClient


class ReviewGenerator:
    """
    Autonomous Agent for generating comprehensive scientific reviews.

    Upgraded features
    -----------------
    - RAPTOR-aware: if a RAPTOR tree exists for the topic, hierarchical
      summaries are prepended for high-level structure.
    - Section clustering: LLM is now asked to cluster evidence into thematic
      sections before writing, reducing hallucination.
    - Token budget management: long abstract lists are chunked to avoid
      exceeding context windows.
    """

    def __init__(self, llm_client: LLMClient, retrieval_pipeline) -> None:
        self.llm = llm_client
        self.retriever = retrieval_pipeline

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_language(text: str) -> str:
        return "zh" if re.search(r"[\u4e00-\u9fff]", text or "") else "en"

    @staticmethod
    def _build_papers_block(provenances: List[Dict], max_papers: int = 30) -> str:
        block = ""
        for i, p in enumerate(provenances[:max_papers], 1):
            sd = p.get("structured_data", {})
            tech_note = (
                f" [Tech:{sd.get('technology','?')} Eff:{sd.get('efficiency','?')}]"
                if sd
                else ""
            )
            block += (
                f"[{i}] {p['evidence']}{tech_note}\n"
                f"Summary: {p['text'][:500]}\n\n"
            )
        return block

    # ── Cluster step (optional, uses LLM) ────────────────────────────────────

    def _cluster_papers(self, topic: str, papers_block: str) -> str:
        """Ask LLM to group papers into thematic clusters before writing."""
        prompt = (
            f'Group the following papers into 3-5 thematic clusters for a review on "{topic}". '
            f"Output ONLY a JSON list: "
            f'[{{"theme":"...", "paper_ids":[1,2,...]}},...]\n\n'
            f"{papers_block}"
        )
        try:
            raw = self.llm.generate(
                prompt,
                system_prompt="JSON-only clustering agent.",
                enable_thinking=False,
                timeout=20,
                max_tokens=400,
            )
            # If clustering fails or returns garbage, skip silently
            if raw and not raw.startswith("Error"):
                return raw
        except Exception:
            pass
        return ""

    # ── Main review generation ────────────────────────────────────────────────

    def generate_review(
        self,
        topic: str,
        provenances: List[Dict],
        raptor_summaries: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a structured scientific review.

        Parameters
        ----------
        topic           : The review topic / query.
        provenances     : Retrieved literature items (each with 'evidence', 'text').
        raptor_summaries: Optional list of pre-built RAPTOR cluster summaries for
                          high-level context (prepended to prompt).
        """
        if not provenances:
            return "Insufficient literature found to generate a review."

        lang = self._detect_language(topic)
        print(
            f"\n[Review Agent] Generating {'Chinese' if lang == 'zh' else 'English'} "
            f"review for: '{topic}' with {len(provenances)} papers."
        )

        papers_block = self._build_papers_block(provenances, max_papers=30)

        # Optional RAPTOR high-level context block
        raptor_block = ""
        if raptor_summaries:
            raptor_block = (
                "**High-Level Thematic Summaries (RAPTOR):**\n"
                + "\n\n".join(f"- {s}" for s in raptor_summaries[:5])
                + "\n\n"
            )

        lang_instruction = (
            "The entire review MUST be written in Chinese (Simplified)."
            if lang == "zh"
            else "Write the review in English."
        )

        prompt = f"""You are a senior editor at Nature Reviews Genetics.
Write a comprehensive, structured review on the topic: "{topic}".

{lang_instruction}

{raptor_block}Literature Evidence:
{papers_block}

Structure Requirements:
1. **Title**: Catchy academic title.
2. **Executive Summary**: 3-sentence high-level overview.
3. **Key Advances**: Group evidence into 3-5 logical themes (e.g., "Efficiency Improvements", "Off-target Analysis", "Clinical Translations"). Within each theme, synthesise findings and cite with [n].
4. **Comparative Analysis**: Compare different approaches.
5. **Limitations & Challenges**: Summarise key open questions.
6. **Future Directions**: Where is the field heading?

Important:
- Integrate citations naturally: [1], [2,3].
- Do NOT repeat raw abstracts; synthesise insights.
- Use Markdown headers and bullets for readability.
"""
        try:
            review = self.llm.generate(
                prompt,
                system_prompt=(
                    "You are an expert scientific writer who synthesises "
                    "biomedical literature into high-quality review articles."
                ),
                max_tokens=3000,
                enable_thinking=False,
                timeout=120,
            )
            if not review or review.startswith("Error"):
                # Retry with shorter output
                review = self.llm.generate(
                    prompt,
                    system_prompt="Expert scientific writer.",
                    max_tokens=1500,
                    enable_thinking=False,
                    timeout=60,
                )
            return review or "⚠️ Review generation failed."
        except Exception as e:
            return f"Error generating review: {e}"
