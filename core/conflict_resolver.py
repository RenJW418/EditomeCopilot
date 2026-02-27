"""
Conflict-Aware Evidence Aggregation  (CAEA)
============================================

Innovation: Biomedical literature frequently contains contradictory findings
(e.g., editing efficiency of "30%" vs "60%" for the same target).  Standard
RAG systems blindly feed all evidence to the LLM, leading to hallucinated
or inconsistent answers.  CAEA detects evidential conflicts, resolves them
using the evidence pyramid + recency, and generates a *structured consensus*
with annotated disagreements.

Mathematical Formulation
------------------------
Given a set of retrieved evidence snippets D = {d_1, …, d_n}:

1. **Claim Extraction** — Group evidence by semantic topic clusters using
   lightweight keyword hashing + overlap scoring:

       sim(d_i, d_j) = |tokens(d_i) ∩ tokens(d_j)| / min(|tokens(d_i)|, |tokens(d_j)|)
       Cluster if sim > θ_cluster (default 0.25)

2. **Conflict Detection** — Within each cluster, detect contradictions:

   a) *Numerical conflicts* — Extract numbers with regex; flag if values
      differ by > 50% for same metric type (efficiency, reduction, etc.)

   b) *Stance conflicts* — Detect opposing sentiment signals:
      positive = {effective, successful, improved, approved}
      negative = {failed, terminated, rejected, adverse, ineffective}
      If both appear in same cluster → conflict.

3. **Conflict Resolution** via Evidence Pyramid:

       winner = argmin_{d ∈ cluster}( evidence_rank(d) )
                                        // lower rank = higher evidence

   Ties broken by recency (newer wins).

4. **Consensus Synthesis** — Output structure:

       {
         "consensus": str,          // resolved view
         "confidence": float,       // 0-1, lower if many conflicts
         "conflicts": [             // annotated disagreements
           {"topic", "claim_a", "claim_b", "resolution", "evidence_basis"}
         ],
         "agreement_ratio": float   // fraction of non-conflicting clusters
       }

Cite as Algorithm 3 in the paper.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from core.evidence_scorer import EVIDENCE_LEVELS, classify_evidence_level, _extract_year


# ─────────────────────────────────────────────────────────────────────────────
# Keyword-based lightweight clustering
# ─────────────────────────────────────────────────────────────────────────────

# Domain-specific stop words to exclude from clustering
_STOP_WORDS: Set[str] = {
    "the", "and", "for", "with", "this", "that", "from", "was", "were",
    "are", "been", "have", "has", "had", "not", "but", "can", "will",
    "also", "than", "these", "their", "its", "our", "using", "used",
    "based", "between", "both", "after", "before", "into", "through",
    "study", "results", "showed", "demonstrated", "revealed", "found",
    "however", "although", "while", "which", "about", "more", "most",
}

# Sentiment / stance signals
_POSITIVE_SIGNALS = re.compile(
    r"\b(?:effective|efficient|success|improved|approv|promising|positive|"
    r"breakthrough|superior|remarkable|significant (?:improvement|reduction|increase)|"
    r"well[- ]tolerat|safe(?:ly)?|durable|sustained|robust)\b", re.IGNORECASE
)
_NEGATIVE_SIGNALS = re.compile(
    r"\b(?:fail|terminat|reject|adverse|ineffective|withdraw|discontinu|"
    r"inferior|unsafe|toxic|lethal|severe|fatal|concern|limitation|"
    r"no (?:significant|improvement|benefit|effect)|unsuccessful|halted)\b", re.IGNORECASE
)

# Numerical metric extraction (efficiency, reduction, rate, etc.)
_NUMBER_METRIC = re.compile(
    r"(\d+(?:\.\d+)?)\s*%\s*"
    r"(?:edit(?:ing)?|efficien|reduc|increase|decrease|knock|induci|express|"
    r"correction|deletio|insertion|survival|response|engraft)?",
    re.IGNORECASE,
)


def _tokenize(text: str) -> Set[str]:
    """Tokenize text into lowered keyword set (remove stop words, keep >= 3 chars)."""
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return {t for t in tokens if t not in _STOP_WORDS}


def _overlap_sim(a: Set[str], b: Set[str]) -> float:
    """Overlap coefficient = |A ∩ B| / min(|A|, |B|)."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


# ─────────────────────────────────────────────────────────────────────────────
# CAEA Engine
# ─────────────────────────────────────────────────────────────────────────────
class ConflictResolver:
    """
    Conflict-Aware Evidence Aggregation engine.

    Parameters
    ----------
    cluster_threshold : float
        Minimum overlap coefficient to group documents (default 0.25).
    numeric_conflict_ratio : float
        Relative difference threshold for numerical conflict (default 0.50).
    enabled : bool
        Master switch (for ablation).
    """

    def __init__(
        self,
        cluster_threshold: float = 0.25,
        numeric_conflict_ratio: float = 0.50,
        enabled: bool = True,
    ):
        self.cluster_threshold = cluster_threshold
        self.numeric_conflict_ratio = numeric_conflict_ratio
        self.enabled = enabled

    def resolve(self, provenance: List[Dict]) -> Dict[str, Any]:
        """
        Detect and resolve conflicts in a provenance list.

        Parameters
        ----------
        provenance : list of dict
            EPARS-scored provenance (should have epars_level, pub_year fields).

        Returns
        -------
        dict with keys:
            - consensus_provenance : list of dict  (re-ordered, conflict-annotated)
            - conflicts : list of dict
            - clusters : list of list of int  (indices)
            - agreement_ratio : float
            - confidence : float
        """
        if not self.enabled or len(provenance) < 2:
            return {
                "consensus_provenance": provenance,
                "conflicts": [],
                "clusters": [[i] for i in range(len(provenance))],
                "agreement_ratio": 1.0,
                "confidence": 1.0,
            }

        # Step 1: Cluster by topic similarity
        clusters = self._cluster(provenance)

        # Step 2: Detect conflicts within each cluster
        all_conflicts: List[Dict] = []
        conflict_clusters = 0

        for cluster_indices in clusters:
            if len(cluster_indices) < 2:
                continue
            cluster_docs = [provenance[i] for i in cluster_indices]
            conflicts = self._detect_conflicts(cluster_docs, cluster_indices)
            if conflicts:
                conflict_clusters += 1
                all_conflicts.extend(conflicts)

        # Step 3: Compute agreement ratio
        n_clusters = max(len(clusters), 1)
        agreement_ratio = 1.0 - (conflict_clusters / n_clusters)

        # Step 4: Confidence score
        #   Confidence drops with more conflicts and lower evidence quality
        conflict_penalty = min(len(all_conflicts) * 0.10, 0.50)
        avg_rank = self._avg_evidence_rank(provenance[:10])
        quality_factor = 1.0 - (avg_rank / 6.0)  # rank 0-5, normalised
        confidence = max(0.1, (0.5 * agreement_ratio + 0.5 * quality_factor) - conflict_penalty)

        # Step 5: Annotate provenance with conflict info
        conflict_doc_indices: Set[int] = set()
        for c in all_conflicts:
            conflict_doc_indices.update(c.get("doc_indices", []))

        annotated = []
        for i, p in enumerate(provenance):
            p = dict(p)
            p["has_conflict"] = i in conflict_doc_indices
            p["conflict_count"] = sum(
                1 for c in all_conflicts if i in c.get("doc_indices", [])
            )
            annotated.append(p)

        return {
            "consensus_provenance": annotated,
            "conflicts": all_conflicts,
            "clusters": clusters,
            "agreement_ratio": round(agreement_ratio, 3),
            "confidence": round(confidence, 3),
        }

    # ── Clustering ──────────────────────────────────────
    def _cluster(self, provenance: List[Dict]) -> List[List[int]]:
        """Single-pass greedy clustering by keyword overlap."""
        token_sets = []
        for p in provenance:
            text = (p.get("evidence", "") or "") + " " + (p.get("text", "") or "")
            token_sets.append(_tokenize(text))

        clusters: List[List[int]] = []
        assigned: Set[int] = set()

        for i in range(len(provenance)):
            if i in assigned:
                continue
            cluster = [i]
            assigned.add(i)
            for j in range(i + 1, len(provenance)):
                if j in assigned:
                    continue
                if _overlap_sim(token_sets[i], token_sets[j]) >= self.cluster_threshold:
                    cluster.append(j)
                    assigned.add(j)
            clusters.append(cluster)

        return clusters

    # ── Conflict Detection ──────────────────────────────
    def _detect_conflicts(
        self, cluster_docs: List[Dict], indices: List[int]
    ) -> List[Dict]:
        """Detect numerical and stance conflicts within a cluster."""
        conflicts: List[Dict] = []

        for i in range(len(cluster_docs)):
            for j in range(i + 1, len(cluster_docs)):
                d_i = cluster_docs[i]
                d_j = cluster_docs[j]
                txt_i = (d_i.get("text", "") or "") + " " + (d_i.get("evidence", "") or "")
                txt_j = (d_j.get("text", "") or "") + " " + (d_j.get("evidence", "") or "")

                # a) Numerical conflict
                nums_i = [float(m) for m in _NUMBER_METRIC.findall(txt_i)]
                nums_j = [float(m) for m in _NUMBER_METRIC.findall(txt_j)]
                if nums_i and nums_j:
                    max_i = max(nums_i)
                    max_j = max(nums_j)
                    if max_i > 0 and max_j > 0:
                        ratio = abs(max_i - max_j) / max(max_i, max_j)
                        if ratio > self.numeric_conflict_ratio:
                            winner, basis = self._resolve_pair(d_i, d_j, indices[i], indices[j])
                            conflicts.append({
                                "type": "numerical",
                                "topic": self._extract_topic(txt_i, txt_j),
                                "claim_a": f"{d_i.get('evidence', '?')}: {max_i:.1f}%",
                                "claim_b": f"{d_j.get('evidence', '?')}: {max_j:.1f}%",
                                "resolution": winner,
                                "evidence_basis": basis,
                                "doc_indices": [indices[i], indices[j]],
                            })

                # b) Stance conflict
                pos_i = bool(_POSITIVE_SIGNALS.search(txt_i))
                neg_i = bool(_NEGATIVE_SIGNALS.search(txt_i))
                pos_j = bool(_POSITIVE_SIGNALS.search(txt_j))
                neg_j = bool(_NEGATIVE_SIGNALS.search(txt_j))

                if (pos_i and neg_j and not neg_i) or (neg_i and pos_j and not pos_i):
                    winner, basis = self._resolve_pair(d_i, d_j, indices[i], indices[j])
                    conflicts.append({
                        "type": "stance",
                        "topic": self._extract_topic(txt_i, txt_j),
                        "claim_a": f"{d_i.get('evidence', '?')}: {'positive' if pos_i else 'negative'}",
                        "claim_b": f"{d_j.get('evidence', '?')}: {'positive' if pos_j else 'negative'}",
                        "resolution": winner,
                        "evidence_basis": basis,
                        "doc_indices": [indices[i], indices[j]],
                    })

        return conflicts

    # ── Conflict Resolution via Evidence Pyramid ────────
    def _resolve_pair(
        self, d_a: Dict, d_b: Dict, idx_a: int, idx_b: int
    ) -> Tuple[str, str]:
        """
        Resolve a conflict between two documents using evidence pyramid + recency.

        Returns (winner_description, resolution_basis).
        """
        rank_a = d_a.get("epars_level_rank", 5)
        rank_b = d_b.get("epars_level_rank", 5)

        year_a = d_a.get("pub_year") or _extract_year(d_a)
        year_b = d_b.get("pub_year") or _extract_year(d_b)

        if rank_a < rank_b:
            return (
                f"Doc #{idx_a + 1} ({d_a.get('epars_level_label', '?')})",
                f"Higher evidence level ({d_a.get('epars_level', '?')} vs {d_b.get('epars_level', '?')})"
            )
        elif rank_b < rank_a:
            return (
                f"Doc #{idx_b + 1} ({d_b.get('epars_level_label', '?')})",
                f"Higher evidence level ({d_b.get('epars_level', '?')} vs {d_a.get('epars_level', '?')})"
            )
        else:
            # Same evidence level → prefer more recent
            if year_a and year_b and year_a != year_b:
                if year_a > year_b:
                    return (
                        f"Doc #{idx_a + 1} (year {year_a})",
                        f"Same evidence level; more recent ({year_a} vs {year_b})"
                    )
                else:
                    return (
                        f"Doc #{idx_b + 1} (year {year_b})",
                        f"Same evidence level; more recent ({year_b} vs {year_a})"
                    )
            return ("Unresolved — equal evidence and recency", "Both claims should be reported")

    # ── Helper: extract cluster topic ──────────────────
    @staticmethod
    def _extract_topic(text_a: str, text_b: str) -> str:
        """Extract 2-3 most common shared tokens as topic label."""
        tokens_a = _tokenize(text_a)
        tokens_b = _tokenize(text_b)
        shared = tokens_a & tokens_b
        if not shared:
            return "general"
        # Sort by length descending (longer words more informative)
        top = sorted(shared, key=len, reverse=True)[:3]
        return " / ".join(top)

    # ── Helper: average evidence rank ──────────────────
    @staticmethod
    def _avg_evidence_rank(provenance: List[Dict]) -> float:
        if not provenance:
            return 5.0
        ranks = [p.get("epars_level_rank", 5) for p in provenance]
        return sum(ranks) / len(ranks)

    # ── Report generation ─────────────────────────────
    @staticmethod
    def format_conflict_report(result: Dict, language: str = "en") -> str:
        """Generate human-readable conflict report for injection into LLM context."""
        conflicts = result.get("conflicts", [])
        agreement = result.get("agreement_ratio", 1.0)
        confidence = result.get("confidence", 1.0)

        if not conflicts:
            if language == "zh":
                return (
                    f"### ✅ 证据一致性报告\n"
                    f"检索到的证据**无明显矛盾** (一致率: {agreement:.0%}, 置信度: {confidence:.2f})"
                )
            return (
                f"### ✅ Evidence Consistency Report\n"
                f"No conflicts detected in retrieved evidence "
                f"(agreement: {agreement:.0%}, confidence: {confidence:.2f})"
            )

        if language == "zh":
            lines = [
                f"### ⚠️ 证据冲突报告",
                f"检测到 **{len(conflicts)} 处冲突** "
                f"(一致率: {agreement:.0%}, 置信度: {confidence:.2f})",
            ]
            for i, c in enumerate(conflicts, 1):
                lines.append(f"\n**冲突 #{i}** [{c['type']}] — 主题: {c['topic']}")
                lines.append(f"  - 观点 A: {c['claim_a']}")
                lines.append(f"  - 观点 B: {c['claim_b']}")
                lines.append(f"  - 🏆 倾向: {c['resolution']}")
                lines.append(f"  - 依据: {c['evidence_basis']}")
        else:
            lines = [
                f"### ⚠️ Evidence Conflict Report",
                f"Detected **{len(conflicts)} conflict(s)** "
                f"(agreement: {agreement:.0%}, confidence: {confidence:.2f})",
            ]
            for i, c in enumerate(conflicts, 1):
                lines.append(f"\n**Conflict #{i}** [{c['type']}] — Topic: {c['topic']}")
                lines.append(f"  - Claim A: {c['claim_a']}")
                lines.append(f"  - Claim B: {c['claim_b']}")
                lines.append(f"  - 🏆 Preferred: {c['resolution']}")
                lines.append(f"  - Basis: {c['evidence_basis']}")

        return "\n".join(lines)
