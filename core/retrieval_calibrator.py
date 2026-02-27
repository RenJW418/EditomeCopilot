"""
Retrieval Confidence Calibration  (RCC)
========================================

Innovation: Standard RAG systems have no notion of *retrieval confidence* —
they return answers regardless of whether suitable evidence was found.
RCC provides a calibrated confidence score that:
  1. Alerts users when evidence is insufficient
  2. Triggers adaptive retrieval strategies (deeper search, HyDE variants)
  3. Annotates the final answer with uncertainty bands

Mathematical Formulation
------------------------
For query *q* with retrieved document set D = {d_1, …, d_n}:

    C(q, D) = λ₁·S_gap + λ₂·S_cov + λ₃·S_div + λ₄·S_qual

Where (all normalised to [0, 1]):

1. **Score Gap Signal** (S_gap) — Separation between top-1 and top-k scores:

       S_gap = (score_1 − score_k) / (score_1 + ε)

   Large gap → top documents are highly relevant → higher confidence.

2. **KG Coverage Signal** (S_cov) — Fraction of query entities with
   KG-grounded evidence:

       S_cov = |E_q ∩ E_D| / |E_q|

   Where E_q = query entities, E_D = entities in retrieved docs.

3. **Evidence Diversity Signal** (S_div) — Shannon entropy of evidence
   levels in top-k documents:

       S_div = H(level_dist) / log(num_levels)

   Diverse evidence types → more comprehensive coverage → higher confidence.

4. **Evidence Quality Signal** (S_qual) — Proportion of top-k documents
   at evidence level III or higher:

       S_qual = |{d ∈ D_topk : rank(d) ≤ 3}| / k

Confidence Thresholds:
    C ≥ 0.70  →  HIGH confidence   (answer directly)
    C ≥ 0.40  →  MODERATE          (answer with caveats)
    C <  0.40  →  LOW              (trigger deeper retrieval + disclaimer)

Adaptive Behaviour (when C < threshold):
    - Retry retrieval with HyDE-augmented variants
    - Expand query depth in KG-AQE
    - Decompose query further
    - Append uncertainty disclaimer to answer

Cite as Algorithm 4 in the paper.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Confidence Thresholds
# ─────────────────────────────────────────────────────────────────────────────
CONFIDENCE_HIGH = 0.70
CONFIDENCE_MODERATE = 0.40

CONFIDENCE_LABELS = {
    "HIGH": {"threshold": CONFIDENCE_HIGH, "emoji": "🟢", "label": "High Confidence"},
    "MODERATE": {"threshold": CONFIDENCE_MODERATE, "emoji": "🟡", "label": "Moderate Confidence"},
    "LOW": {"threshold": 0.0, "emoji": "🔴", "label": "Low Confidence"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Signal Computers
# ─────────────────────────────────────────────────────────────────────────────

def _score_gap_signal(
    provenance: List[Dict],
    top_k: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    S_gap: measures how much the top document stands out.
    Large gap → clear relevance signal → higher confidence.
    """
    scores = [float(p.get("score", 0.0)) for p in provenance[:top_k]]
    if len(scores) < 2:
        return 0.0
    s1 = scores[0]
    sk = scores[-1]
    return (s1 - sk) / (s1 + eps)


def _kg_coverage_signal(
    query: str,
    provenance: List[Dict],
    kg_entities: Optional[Set[str]] = None,
) -> float:
    """
    S_cov: fraction of query entities that appear in retrieved evidence.
    """
    # Extract entity-like tokens from query (capitalised words, known abbreviations)
    q_entities = set(re.findall(r"\b[A-Z][A-Za-z0-9]{2,}\b", query))
    q_entities.update(
        re.findall(r"\b(?:CRISPR|Cas\d+[a-z]?|ABE|CBE|PE\d?|LNP|AAV\d?|RNP)\b", query, re.IGNORECASE)
    )
    q_entities = {e.upper() for e in q_entities}

    if not q_entities:
        return 1.0  # No specific entities → coverage not applicable

    # Check presence in retrieved evidence
    all_text = " ".join(
        (p.get("text", "") or "") + " " + (p.get("evidence", "") or "")
        for p in provenance[:15]
    ).upper()

    found = sum(1 for e in q_entities if e in all_text)
    coverage = found / len(q_entities) if q_entities else 1.0

    # Bonus if KG entities are provided (from KG-AQE linked_entities)
    if kg_entities:
        kg_found = sum(1 for e in q_entities if e in {k.upper() for k in kg_entities})
        kg_ratio = kg_found / len(q_entities) if q_entities else 1.0
        coverage = 0.7 * coverage + 0.3 * kg_ratio

    return min(coverage, 1.0)


def _evidence_diversity_signal(
    provenance: List[Dict],
    top_k: int = 15,
) -> float:
    """
    S_div: Shannon entropy of evidence level distribution, normalised.
    Diverse evidence types → more comprehensive → higher confidence.
    """
    levels = [p.get("epars_level", "LEVEL_V") for p in provenance[:top_k]]
    if not levels:
        return 0.0

    # Count distribution
    dist: Dict[str, int] = {}
    for lvl in levels:
        dist[lvl] = dist.get(lvl, 0) + 1

    n = len(levels)
    num_classes = len(dist)
    if num_classes <= 1:
        return 0.0

    # Shannon entropy
    entropy = 0.0
    for count in dist.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)

    # Normalise by max entropy (uniform distribution)
    max_entropy = math.log2(min(num_classes, 6))  # 6 evidence levels
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _evidence_quality_signal(
    provenance: List[Dict],
    top_k: int = 10,
    quality_threshold_rank: int = 3,  # Level III or higher
) -> float:
    """
    S_qual: proportion of top-k documents at evidence level ≤ III.
    More high-level evidence → higher confidence.
    """
    if not provenance:
        return 0.0
    docs = provenance[:top_k]
    high_quality = sum(
        1 for p in docs
        if p.get("epars_level_rank", 5) <= quality_threshold_rank
    )
    return high_quality / len(docs)


# ─────────────────────────────────────────────────────────────────────────────
# RCC Engine
# ─────────────────────────────────────────────────────────────────────────────
class RetrievalCalibrator:
    """
    Retrieval Confidence Calibration engine.

    Parameters
    ----------
    lambda_gap : float
        Weight for score gap signal (default 0.25).
    lambda_cov : float
        Weight for KG coverage signal (default 0.25).
    lambda_div : float
        Weight for evidence diversity signal (default 0.25).
    lambda_qual : float
        Weight for evidence quality signal (default 0.25).
    enabled : bool
        Master switch (for ablation).
    """

    def __init__(
        self,
        lambda_gap: float = 0.25,
        lambda_cov: float = 0.25,
        lambda_div: float = 0.25,
        lambda_qual: float = 0.25,
        enabled: bool = True,
    ):
        self.lambda_gap = lambda_gap
        self.lambda_cov = lambda_cov
        self.lambda_div = lambda_div
        self.lambda_qual = lambda_qual
        self.enabled = enabled

    def calibrate(
        self,
        query: str,
        provenance: List[Dict],
        kg_entities: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute retrieval confidence for a query + provenance set.

        Parameters
        ----------
        query : str
            The user query.
        provenance : list of dict
            EPARS-scored provenance (should have epars_level, epars_level_rank).
        kg_entities : set of str, optional
            KG-linked entity IDs from KG-AQE (for coverage computation).

        Returns
        -------
        dict with keys:
            - confidence : float (0-1)
            - confidence_label : str (HIGH / MODERATE / LOW)
            - signals : dict of individual signal values
            - needs_deeper_retrieval : bool
            - uncertainty_disclaimer : str or None
        """
        if not self.enabled or not provenance:
            return {
                "confidence": 0.5,
                "confidence_label": "MODERATE",
                "signals": {},
                "needs_deeper_retrieval": False,
                "uncertainty_disclaimer": None,
            }

        # Compute individual signals
        s_gap = _score_gap_signal(provenance)
        s_cov = _kg_coverage_signal(query, provenance, kg_entities)
        s_div = _evidence_diversity_signal(provenance)
        s_qual = _evidence_quality_signal(provenance)

        # Weighted combination
        confidence = (
            self.lambda_gap * s_gap
            + self.lambda_cov * s_cov
            + self.lambda_div * s_div
            + self.lambda_qual * s_qual
        )
        confidence = max(0.0, min(1.0, confidence))

        # Classify confidence level
        if confidence >= CONFIDENCE_HIGH:
            label = "HIGH"
        elif confidence >= CONFIDENCE_MODERATE:
            label = "MODERATE"
        else:
            label = "LOW"

        # Determine if deeper retrieval is needed
        needs_deeper = label == "LOW"

        # Generate disclaimer text
        disclaimer = None
        if label == "LOW":
            disclaimer = (
                "⚠️ **Low retrieval confidence** — The knowledge base may not "
                "contain sufficient evidence for this query. Findings should be "
                "interpreted with caution and verified against primary literature."
            )
        elif label == "MODERATE":
            disclaimer = (
                "ℹ️ **Moderate confidence** — Some aspects of this query are "
                "well-covered but evidence may be incomplete for certain sub-topics."
            )

        return {
            "confidence": round(confidence, 3),
            "confidence_label": label,
            "signals": {
                "score_gap": round(s_gap, 4),
                "kg_coverage": round(s_cov, 4),
                "evidence_diversity": round(s_div, 4),
                "evidence_quality": round(s_qual, 4),
            },
            "needs_deeper_retrieval": needs_deeper,
            "uncertainty_disclaimer": disclaimer,
        }

    # ── Report generation ─────────────────────────────
    @staticmethod
    def format_confidence_report(result: Dict, language: str = "en") -> str:
        """Generate a human-readable confidence report."""
        conf = result.get("confidence", 0)
        label = result.get("confidence_label", "?")
        signals = result.get("signals", {})
        info = CONFIDENCE_LABELS.get(label, {})
        emoji = info.get("emoji", "❓")

        if language == "zh":
            label_zh = {"HIGH": "高", "MODERATE": "中", "LOW": "低"}.get(label, "?")
            lines = [
                f"### {emoji} 检索置信度: {label_zh} ({conf:.2f})",
                f"- 得分差异信号 (S_gap): {signals.get('score_gap', 0):.3f}",
                f"- 知识覆盖信号 (S_cov): {signals.get('kg_coverage', 0):.3f}",
                f"- 证据多样性信号 (S_div): {signals.get('evidence_diversity', 0):.3f}",
                f"- 证据质量信号 (S_qual): {signals.get('evidence_quality', 0):.3f}",
            ]
        else:
            lines = [
                f"### {emoji} Retrieval Confidence: {info.get('label', label)} ({conf:.2f})",
                f"- Score Gap (S_gap): {signals.get('score_gap', 0):.3f}",
                f"- KG Coverage (S_cov): {signals.get('kg_coverage', 0):.3f}",
                f"- Evidence Diversity (S_div): {signals.get('evidence_diversity', 0):.3f}",
                f"- Evidence Quality (S_qual): {signals.get('evidence_quality', 0):.3f}",
            ]

        if result.get("needs_deeper_retrieval"):
            if language == "zh":
                lines.append("**→ 已触发深度检索以补充证据**")
            else:
                lines.append("**→ Deeper retrieval triggered to supplement evidence**")

        return "\n".join(lines)

    @staticmethod
    def inject_uncertainty(
        response: str,
        calibration: Dict,
        language: str = "en",
    ) -> str:
        """Append uncertainty disclaimer to the response if needed."""
        disclaimer = calibration.get("uncertainty_disclaimer")
        if disclaimer:
            conf = calibration.get("confidence", 0)
            label = calibration.get("confidence_label", "?")
            info = CONFIDENCE_LABELS.get(label, {})
            emoji = info.get("emoji", "")

            if language == "zh":
                label_zh = {"HIGH": "高", "MODERATE": "中", "LOW": "低"}.get(label, "?")
                footer = (
                    f"\n\n---\n"
                    f"{emoji} **检索置信度: {label_zh}** ({conf:.2f})\n\n"
                )
                if label == "LOW":
                    footer += (
                        "⚠️ **低置信度警告** — 知识库中可能缺乏该查询的充分证据，"
                        "请谨慎解读以上结论并参考原始文献。"
                    )
                elif label == "MODERATE":
                    footer += (
                        "ℹ️ **中置信度提示** — 部分子问题的证据覆盖可能不完整。"
                    )
            else:
                footer = (
                    f"\n\n---\n"
                    f"{emoji} **Retrieval Confidence: {info.get('label', label)}** ({conf:.2f})\n\n"
                    f"{disclaimer}"
                )
            return response + footer
        return response
