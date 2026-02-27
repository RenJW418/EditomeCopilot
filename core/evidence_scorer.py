"""
Evidence-Pyramid-Aware Retrieval Scoring  (EPARS)
=================================================

Innovation: Standard RAG treats every retrieved chunk as equally reliable.
In biomedicine, evidence quality varies enormously — a Phase-III RCT outweighs
an in-vitro report by orders of magnitude.  EPARS adds *domain-calibrated*
evidence-level scoring to the retrieval pipeline.

Mathematical Formulation
------------------------
For a document *d* retrieved for query *q*, the final score is:

    S_final(d, q) = S_rel(d,q)^α  ×  W_evi(d)  ×  D_rec(d)

Where:
    S_rel  — cross-encoder relevance score (already computed upstream)
    α      — relevance exponent (default 0.7, dampens very-high/low scores)
    W_evi  — evidence-level weight mapped via the biomedical evidence pyramid:
               Level I   (systematic review / meta-analysis)      → 1.40
               Level II  (RCT)                                    → 1.30
               Level III (controlled trial / cohort)              → 1.15
               Level IV  (case-control / case series)             → 1.00
               Level V   (expert opinion / in-vitro)              → 0.85
               FDA/EMA approval                                   → 1.50
    D_rec  — temporal recency decay:
               D_rec(d) = β + (1-β) × exp(−λ × Δt)
             where Δt = current_year − pub_year, β=0.5 clamp, λ=0.08

The three multiplicative factors capture *what you said* (relevance),
*how trustworthy it is* (evidence), and *how current it is* (recency).

Cite as Algorithm 1 in the paper.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Evidence-Level Taxonomy & Weights
# ─────────────────────────────────────────────────────────────────────────────
EVIDENCE_LEVELS: Dict[str, Dict[str, Any]] = {
    "APPROVED": {
        "label": "Regulatory Approval (FDA/EMA/MHRA)",
        "weight": 1.50,
        "rank": 0,
        "description": "Therapy has received regulatory approval",
    },
    "LEVEL_I": {
        "label": "Systematic Review / Meta-analysis",
        "weight": 1.40,
        "rank": 1,
        "description": "Systematic review or meta-analysis of RCTs",
    },
    "LEVEL_II": {
        "label": "Randomised Controlled Trial",
        "weight": 1.30,
        "rank": 2,
        "description": "Well-designed RCT, Phase II/III clinical trial",
    },
    "LEVEL_III": {
        "label": "Controlled Trial / Prospective Cohort",
        "weight": 1.15,
        "rank": 3,
        "description": "Non-randomised controlled study, Phase I trial, cohort study",
    },
    "LEVEL_IV": {
        "label": "Case-Control / Case Series / Retrospective",
        "weight": 1.00,
        "rank": 4,
        "description": "Case-control study, case report, retrospective analysis",
    },
    "LEVEL_V": {
        "label": "Expert Opinion / In-vitro / Computational",
        "weight": 0.85,
        "rank": 5,
        "description": "Expert opinion, in-vitro, cell line study, computational prediction",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Regex-based Evidence-Level Classifier
# ─────────────────────────────────────────────────────────────────────────────
_LEVEL_PATTERNS: List[Tuple[str, str]] = [
    # Regulatory approval markers
    ("APPROVED", r"\b(?:FDA[- ]?approv|EMA[- ]?approv|MHRA[- ]?approv|regulatory approval"
                 r"|market(?:ing)? authoriz|conditionally approved)\b"),
    # Level I — meta-analysis / systematic review
    ("LEVEL_I",  r"\b(?:meta[- ]?analy|systematic review|cochrane|pooled analysis"
                 r"|umbrella review|network meta)\b"),
    # Level II — RCT / Phase II-III
    ("LEVEL_II", r"\b(?:randomi[sz]ed controlled|phase\s*(?:II[Ib]?|III|2[bB]?|3)"
                 r"|double[- ]blind|placebo[- ]controlled|multicent(?:er|re)\s+trial"
                 r"|pivotal trial|registrational)\b"),
    # Level III — Phase I / controlled / cohort
    ("LEVEL_III", r"\b(?:phase\s*(?:I[ab]?|1[ab]?)|prospective cohort|dose[- ]?escalation"
                  r"|first[- ]?in[- ]?human|open[- ]?label|non[- ]?randomi[sz]ed"
                  r"|controlled study|longitudinal study)\b"),
    # Level IV — case / retrospective
    ("LEVEL_IV", r"\b(?:case[- ]?(?:report|series|control|study)|retrospective"
                 r"|observational study|chart review|single[- ]?center"
                 r"|single[- ]?patient|compassionate use|expanded access)\b"),
    # Level V — in vitro / computational / expert
    ("LEVEL_V",  r"\b(?:in[- ]?vitro|cell[- ]?line|HEK293|organoid|in silico"
                 r"|computational|bioinformatic|molecular dynamic|prediction model"
                 r"|expert (?:opinion|consensus)|review article|narrative review)\b"),
]

# Compiled for speed — applied to every retrieved chunk
_COMPILED_PATTERNS = [(lvl, re.compile(pat, re.IGNORECASE)) for lvl, pat in _LEVEL_PATTERNS]


def classify_evidence_level(text: str) -> str:
    """
    Classify a text snippet into an evidence-pyramid level using
    a cascading regex classifier.  Returns the *highest* level matched.

    Parameters
    ----------
    text : str
        Concatenated title + abstract + metadata text.

    Returns
    -------
    str
        One of 'APPROVED', 'LEVEL_I' … 'LEVEL_V'. Defaults to 'LEVEL_V'
        when no clear signal is found.
    """
    for level_key, pattern in _COMPILED_PATTERNS:
        if pattern.search(text):
            return level_key
    return "LEVEL_V"


# ─────────────────────────────────────────────────────────────────────────────
# Recency Decay Function
# ─────────────────────────────────────────────────────────────────────────────
_CURRENT_YEAR = datetime.now().year

def _extract_year(prov: Dict) -> Optional[int]:
    """Extract publication year from a provenance dict."""
    # Try structured_data.year first
    sd = prov.get("structured_data", {}) or {}
    for key in ("year", "pub_year", "publication_year"):
        y = sd.get(key)
        if y:
            try:
                return int(y)
            except (ValueError, TypeError):
                pass
    # Fallback: scan evidence / text for 4-digit year (19xx–20xx)
    for field in ("evidence", "text"):
        content = prov.get(field, "") or ""
        years = re.findall(r"\b(19[89]\d|20[0-2]\d)\b", content)
        if years:
            return max(int(y) for y in years)
    return None


def recency_decay(year: Optional[int], beta: float = 0.50, lam: float = 0.08) -> float:
    """
    Compute temporal recency factor.

        D_rec(d) = β + (1 − β) × exp(−λ × Δt)

    Ensures a minimum weight of β for very old documents (knowledge doesn't
    fully expire) while giving a boost to recent publications.

    Parameters
    ----------
    year : int or None
        Publication year.  None → returns 1.0 (neutral).
    beta : float
        Floor weight (default 0.50).
    lam : float
        Decay rate (default 0.08 ≈ half-weight at ~8.7 years old).

    Returns
    -------
    float
        Value in [beta, 1.0].
    """
    if year is None:
        return 1.0
    delta_t = max(0, _CURRENT_YEAR - year)
    return beta + (1.0 - beta) * math.exp(-lam * delta_t)


# ─────────────────────────────────────────────────────────────────────────────
# EPARS Re-Scoring Engine
# ─────────────────────────────────────────────────────────────────────────────
class EvidencePyramidScorer:
    """
    Evidence-Pyramid-Aware Retrieval Scorer.

    Applies the EPARS formula to a list of provenance dicts:

        S_final = S_rel^α  ×  W_evi  ×  D_rec

    After scoring, documents are re-sorted by S_final.
    """

    def __init__(
        self,
        alpha: float = 0.70,
        beta: float = 0.50,
        lam: float = 0.08,
        enabled: bool = True,
    ):
        """
        Parameters
        ----------
        alpha : float
            Relevance exponent (default 0.70 — dampens extremes).
        beta : float
            Recency floor (default 0.50).
        lam : float
            Recency decay rate.
        enabled : bool
            Master switch for ablation studies.
        """
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.enabled = enabled

    def score(self, provenance: List[Dict], top_k: int = 25) -> List[Dict]:
        """
        Re-score and re-sort a provenance list.

        Each dict is annotated with:
            - epars_level      : str   (evidence level key)
            - epars_level_label: str   (human-readable label)
            - epars_w_evi      : float (evidence weight)
            - epars_d_rec      : float (recency factor)
            - epars_score      : float (final EPARS score)
        The original 'score' field is preserved as 'score_relevance'.

        Returns
        -------
        list of dict
            Re-sorted provenance (highest EPARS first), trimmed to top_k.
        """
        if not self.enabled or not provenance:
            return provenance[:top_k]

        scored: List[Dict] = []
        for p in provenance:
            p = dict(p)  # shallow copy
            text = (p.get("evidence", "") or "") + " " + (p.get("text", "") or "")

            # Classify evidence level
            level = classify_evidence_level(text)
            level_info = EVIDENCE_LEVELS[level]

            # Compute components
            s_rel = max(float(p.get("score", 0.0)), 1e-6)
            w_evi = level_info["weight"]
            year = _extract_year(p)
            d_rec = recency_decay(year, self.beta, self.lam)

            # EPARS formula
            s_final = (s_rel ** self.alpha) * w_evi * d_rec

            # Annotate
            p["score_relevance"] = s_rel
            p["epars_level"] = level
            p["epars_level_label"] = level_info["label"]
            p["epars_level_rank"] = level_info["rank"]
            p["epars_w_evi"] = round(w_evi, 3)
            p["epars_d_rec"] = round(d_rec, 3)
            p["epars_score"] = round(s_final, 6)
            p["score"] = round(s_final, 6)  # replace top-level score
            p["pub_year"] = year

            scored.append(p)

        # Sort by EPARS score descending
        scored.sort(key=lambda x: x["epars_score"], reverse=True)
        return scored[:top_k]

    # ── Analytics helpers (for paper tables / ablation) ────────────
    @staticmethod
    def level_distribution(provenance: List[Dict]) -> Dict[str, int]:
        """Count documents per evidence level (after scoring)."""
        dist: Dict[str, int] = {}
        for p in provenance:
            lvl = p.get("epars_level", "UNKNOWN")
            dist[lvl] = dist.get(lvl, 0) + 1
        return dist

    @staticmethod
    def format_scoring_report(provenance: List[Dict], top_n: int = 5) -> str:
        """Generate a human-readable scoring summary for debugging / paper."""
        lines = ["### 📊 EPARS Evidence Scoring Report"]
        dist = EvidencePyramidScorer.level_distribution(provenance)
        lines.append(f"**Evidence distribution** ({len(provenance)} docs):")
        for lvl, info in EVIDENCE_LEVELS.items():
            cnt = dist.get(lvl, 0)
            if cnt > 0:
                lines.append(f"  - {info['label']}: **{cnt}** ({100*cnt/len(provenance):.0f}%)")
        lines.append(f"\n**Top-{top_n} documents by EPARS score:**")
        for i, p in enumerate(provenance[:top_n], 1):
            lines.append(
                f"  {i}. [{p.get('epars_level_label', '?')}] "
                f"S={p.get('epars_score', 0):.4f} "
                f"(rel={p.get('score_relevance', 0):.4f}, "
                f"evi={p.get('epars_w_evi', 0):.2f}, "
                f"rec={p.get('epars_d_rec', 0):.2f}, "
                f"year={p.get('pub_year', '?')})\n"
                f"     {(p.get('evidence', '') or '')[:80]}"
            )
        return "\n".join(lines)
