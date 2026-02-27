"""
Cross-Species Translation Intelligence
=======================================
Mouse → Human efficiency translation matrix for gene editing.

Core problem: Editing efficiencies reported in mouse models are often dramatically
different from human outcomes. This module provides literature-sourced translation
coefficients to help researchers & clinicians set realistic expectations.

Data: Published mouse→human comparisons from clinical trials vs. preclinical data.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Translation Coefficient Database
# Keys: (technology, tissue, delivery_method)
# Values: mouse→human ratio, actual numbers, references
# ─────────────────────────────────────────────────────────────────────────────
TRANSLATION_MATRIX: List[Dict[str, Any]] = [
    # ── Liver In Vivo ───────────────────────────────────────────────────
    {
        "technology": "CRISPR-Cas9 KO",
        "tissue": "liver",
        "delivery": "LNP",
        "target": "TTR",
        "mouse_efficiency": "97% TTR KD (single dose, C57BL/6)",
        "human_efficiency": "87% TTR KD (single dose, Phase I NTLA-2001)",
        "translation_ratio": 0.90,
        "translation_quality": "HIGH",
        "notes": "Remarkably good translation. LNP liver tropism well-conserved across species. Mouse used higher mg/kg dose.",
        "ref": "10.1056/NEJMoa2107454",
        "year": 2021,
    },
    {
        "technology": "ABE",
        "tissue": "liver",
        "delivery": "LNP",
        "target": "PCSK9",
        "mouse_efficiency": "~70% base conversion (C57BL/6, 1.0 mg/kg)",
        "human_efficiency": "~55% PCSK9 reduction (Phase Ib, 0.45 mg/kg), up to 84% at 0.6 mg/kg",
        "translation_ratio": 0.75,
        "translation_quality": "HIGH",
        "notes": "Good translation for liver editing. Dose-limiting hepatotoxicity at higher doses in human. NHP intermediate: 63% at 1.0 mg/kg.",
        "ref": "10.1056/NEJMoa2303223",
        "year": 2023,
    },
    {
        "technology": "ABE",
        "tissue": "liver",
        "delivery": "LNP",
        "target": "PCSK9 (NHP comparison)",
        "mouse_efficiency": "~70% base conversion (1.0 mg/kg)",
        "human_efficiency": "~63% PCSK9 reduction (NHP cynomolgus, 1.0 mg/kg)",
        "translation_ratio": 0.85,
        "translation_quality": "HIGH",
        "notes": "Mouse→NHP translation for liver base editing. NHP is a better predictor of human liver editing than mouse.",
        "ref": "10.1038/s41586-021-03534-y",
        "year": 2021,
    },
    # ── HSC Ex Vivo ─────────────────────────────────────────────────────
    {
        "technology": "CRISPR-Cas9 KO",
        "tissue": "HSC (ex vivo)",
        "delivery": "RNP electroporation",
        "target": "BCL11A enhancer",
        "mouse_efficiency": "95% indels in mouse HSCs; 70% HbF induction in transplanted mice",
        "human_efficiency": "95% indels in human HSCs; 95% HbF after transplant (Casgevy CLIMB-121)",
        "translation_ratio": 0.95,
        "translation_quality": "HIGH",
        "notes": "Excellent translation for ex vivo RNP editing. Key difference: human HSC engraftment is the bottleneck, not editing efficiency per se.",
        "ref": "10.1056/NEJMoa2031054",
        "year": 2023,
    },
    {
        "technology": "ABE",
        "tissue": "HSC (ex vivo)",
        "delivery": "mRNA electroporation",
        "target": "HBB E6V (SCD correction)",
        "mouse_efficiency": "80% A→G conversion in mouse HSC (sgRNA optimized)",
        "human_efficiency": "58-68% A→G conversion in human CD34+ HSCs",
        "translation_ratio": 0.78,
        "translation_quality": "MEDIUM-HIGH",
        "notes": "Good translation. Human HSCs slightly less permissive to ABE than murine HSCs. HSC viability post-editing is comparable.",
        "ref": "10.1038/s41586-021-03609-w",
        "year": 2021,
    },
    # ── Retina In Vivo ──────────────────────────────────────────────────
    {
        "technology": "CRISPR-Cas9 (dual-guide)",
        "tissue": "retina",
        "delivery": "AAV5 subretinal",
        "target": "CEP290 IVS26",
        "mouse_efficiency": "40% photoreceptor transduction; significant visual rescue in rd16 mice",
        "human_efficiency": "Modest visual improvement in some patients (EDIT-101 Phase I/II); limited dose-response",
        "translation_ratio": 0.3,
        "translation_quality": "LOW",
        "notes": "Poor translation. Mouse retina is much thinner → better viral penetration. Human photoreceptor layer is deeper. AAV5 tropism differs between species.",
        "ref": "10.1056/NEJMoa2309915",
        "year": 2024,
    },
    # ── Muscle In Vivo ──────────────────────────────────────────────────
    {
        "technology": "CRISPR-Cas9 (exon deletion)",
        "tissue": "muscle",
        "delivery": "AAV9 systemic",
        "target": "DMD exon 51",
        "mouse_efficiency": "50-70% dystrophin restoration in mdx mice (high-dose AAV9)",
        "human_efficiency": "~6% dystrophin restoration (Sarepta SRP-9001 Phase I; measured by Western blot)",
        "translation_ratio": 0.10,
        "translation_quality": "LOW",
        "notes": "Very poor translation. Mouse muscle is ~1000x less mass. AAV9 dose (5e14 vg/kg in mouse) cannot be replicated in human due to manufacturing and toxicity limits.",
        "ref": "10.1126/science.aad5143",
        "year": 2016,
    },
    {
        "technology": "ABE",
        "tissue": "muscle",
        "delivery": "AAV9 systemic",
        "target": "RYR1 (pilot study)",
        "mouse_efficiency": "~35% base conversion in mouse quadriceps",
        "human_efficiency": "Not yet tested in human (preclinical only)",
        "translation_ratio": None,  # Predicted: 0.05-0.15 based on DMD AAV9 data
        "translation_quality": "PREDICTED",
        "notes": "No clinical data. Based on DMD AAV9 translation, expect 5-15% of mouse efficiency. Body mass scaling is major challenge.",
        "ref": "Preclinical estimates",
        "year": 2024,
    },
    # ── Brain In Vivo ───────────────────────────────────────────────────
    {
        "technology": "CRISPR-Cas9 KO",
        "tissue": "brain (CNS)",
        "delivery": "AAV.PHP.eB (mouse-specific) / AAV9 intrathecal (human)",
        "target": "Various neuronal targets",
        "mouse_efficiency": "80-95% transduction with AAV.PHP.eB IV in mouse CNS",
        "human_efficiency": "AAV.PHP.eB does NOT cross human BBB (receptor Ly6a is mouse-specific). AAV9 intrathecal: ~10-30% motor neuron transduction (SMA data).",
        "translation_ratio": 0.05,
        "translation_quality": "VERY LOW (IV route), MEDIUM (intrathecal)",
        "notes": "CRITICAL: AAV.PHP.eB BBB-crossing is mouse-specific! Many preclinical CNS papers use this capsid — results do NOT translate to human. Must use intrathecal/intracisternal delivery in human.",
        "ref": "10.1038/s41587-019-0245-7",
        "year": 2019,
    },
    # ── T Cell Ex Vivo ──────────────────────────────────────────────────
    {
        "technology": "CRISPR-Cas9 (multiplex KO)",
        "tissue": "T cells (ex vivo)",
        "delivery": "RNP electroporation",
        "target": "TRAC + B2M + PD-1 (allogeneic CAR-T)",
        "mouse_efficiency": "95% KO per gene; >80% triple-KO in mouse T cells",
        "human_efficiency": "85-95% KO per gene; ~70% triple-KO in human T cells (CTX110 Phase I data)",
        "translation_ratio": 0.85,
        "translation_quality": "HIGH",
        "notes": "Good translation for ex vivo T cell editing. Human T cells slightly more resistant to electroporation. Key concern: translocation risk with multiplex DSBs.",
        "ref": "10.1126/science.abq1441",
        "year": 2022,
    },
    # ── Lung In Vivo ────────────────────────────────────────────────────
    {
        "technology": "ABE",
        "tissue": "lung (epithelium)",
        "delivery": "LNP nebulization / intratracheal",
        "target": "CFTR (CF model)",
        "mouse_efficiency": "15-25% base conversion in airway epithelium (intratracheal LNP)",
        "human_efficiency": "Not yet tested in human. NHP (cynomolgus): 5-12% in conducting airways.",
        "translation_ratio": None,  # Predicted: 0.3-0.5 based on NHP data
        "translation_quality": "PREDICTED",
        "notes": "Lung delivery is challenging. Mucus barrier reduces LNP penetration in CF lungs. Mouse airways lack the thick mucus layer of CF patients.",
        "ref": "10.1038/s41587-023-02004-y",
        "year": 2023,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Tissue-level summary statistics
# ─────────────────────────────────────────────────────────────────────────────
TISSUE_TRANSLATION_SUMMARY: Dict[str, Dict[str, Any]] = {
    "liver": {
        "avg_ratio": 0.83,
        "range": "0.75-0.90",
        "quality": "HIGH",
        "note": "Liver is the best-translated tissue due to LNP natural tropism. NHP is excellent predictor.",
    },
    "HSC (ex vivo)": {
        "avg_ratio": 0.87,
        "range": "0.78-0.95",
        "quality": "HIGH",
        "note": "Ex vivo editing translates well. RNP electroporation is cell-autonomous (no delivery barrier). HSC engraftment is the clinical bottleneck.",
    },
    "T cells (ex vivo)": {
        "avg_ratio": 0.85,
        "range": "0.80-0.95",
        "quality": "HIGH",
        "note": "Excellent translation. T cells are robust and edit efficiently ex vivo across species.",
    },
    "retina": {
        "avg_ratio": 0.30,
        "range": "0.20-0.40",
        "quality": "LOW-MEDIUM",
        "note": "Poor translation due to retinal anatomy differences. Mouse retina is thinner with better viral access.",
    },
    "muscle": {
        "avg_ratio": 0.10,
        "range": "0.05-0.15",
        "quality": "LOW",
        "note": "Very poor translation. Body mass scaling: human muscle is ~1000x mouse. Systemic AAV dose cannot scale linearly.",
    },
    "brain (CNS)": {
        "avg_ratio": 0.05,
        "range": "0.01-0.30 (route-dependent)",
        "quality": "VERY LOW (IV) to MEDIUM (intrathecal)",
        "note": "CRITICAL: AAV.PHP.eB does NOT cross human BBB. Mouse CNS data misleading. Intrathecal delivery required in human.",
    },
    "lung": {
        "avg_ratio": 0.35,
        "range": "0.20-0.50",
        "quality": "LOW-MEDIUM",
        "note": "Mucus barrier in CF patients. Mouse airways lack equivalent obstruction. NHP is better predictor.",
    },
}


class CrossSpeciesTranslator:
    """Translate preclinical editing efficiencies to predicted human outcomes."""

    def __init__(self):
        self.matrix = TRANSLATION_MATRIX
        self.tissue_summary = TISSUE_TRANSLATION_SUMMARY

    def predict_human_efficiency(self, technology: str, tissue: str,
                                  mouse_efficiency_pct: float = None,
                                  delivery: str = None) -> Dict[str, Any]:
        """
        Given a technology+tissue, return predicted human efficiency.

        Args:
            technology: e.g., "ABE", "CRISPR-Cas9", "Prime Editing"
            tissue: e.g., "liver", "muscle", "HSC", "brain"
            mouse_efficiency_pct: Optional float (e.g., 70.0 for 70%)
            delivery: Optional delivery method filter

        Returns:
            Dict with prediction, confidence, references
        """
        tissue_lower = tissue.lower()
        tech_lower = technology.lower()

        # Find matching entries
        matches = []
        for entry in self.matrix:
            e_tissue = entry["tissue"].lower()
            e_tech = entry["technology"].lower()

            tissue_match = any(t in e_tissue for t in tissue_lower.split())
            tech_match = any(t in e_tech for t in tech_lower.split() if len(t) > 2)

            if tissue_match and tech_match:
                matches.append(entry)
            elif tissue_match and not tech_match:
                matches.append(entry)  # Tissue-level fallback

        # Get tissue summary
        tissue_key = None
        for k in self.tissue_summary:
            if any(t in k.lower() for t in tissue_lower.split()):
                tissue_key = k
                break

        summary = self.tissue_summary.get(tissue_key, {})

        result = {
            "technology": technology,
            "tissue": tissue,
            "matched_entries": len(matches),
            "tissue_avg_ratio": summary.get("avg_ratio"),
            "tissue_range": summary.get("range"),
            "quality": summary.get("quality", "UNKNOWN"),
            "note": summary.get("note", ""),
            "specific_matches": [],
        }

        for m in matches[:3]:
            entry_result = {
                "target": m.get("target"),
                "mouse_efficiency": m.get("mouse_efficiency"),
                "human_efficiency": m.get("human_efficiency"),
                "ratio": m.get("translation_ratio"),
                "ref": m.get("ref"),
            }
            result["specific_matches"].append(entry_result)

        # Calculate predicted human efficiency
        if mouse_efficiency_pct and summary.get("avg_ratio"):
            predicted = mouse_efficiency_pct * summary["avg_ratio"]
            result["predicted_human_efficiency_pct"] = round(predicted, 1)
            result["prediction_basis"] = f"Mouse {mouse_efficiency_pct}% × tissue ratio {summary['avg_ratio']} = {predicted:.1f}%"

        return result

    def get_critical_warnings(self, tissue: str) -> List[str]:
        """Return critical cross-species translation warnings for a tissue."""
        warnings = []
        tissue_lower = tissue.lower()

        if "brain" in tissue_lower or "cns" in tissue_lower or "neuron" in tissue_lower:
            warnings.append(
                "🚨 CRITICAL: AAV.PHP.eB does NOT cross human BBB (Ly6a receptor is mouse-specific). "
                "Mouse CNS transduction data using this capsid is NOT translatable to human."
            )

        if "muscle" in tissue_lower:
            warnings.append(
                "⚠️ Mouse→Human muscle translation is ~10%. Body mass scaling means AAV doses "
                "that work in mouse (5e14 vg/kg) cannot be replicated in human due to "
                "manufacturing limits and hepatotoxicity."
            )

        if "retina" in tissue_lower or "eye" in tissue_lower:
            warnings.append(
                "⚠️ Mouse retina is ~5x thinner than human. Subretinal AAV injections cover "
                "proportionally more photoreceptors in mouse. Expect 30-40% of mouse efficacy."
            )

        if "lung" in tissue_lower:
            warnings.append(
                "⚠️ Mouse airways lack the thick mucus layer of human CF patients. "
                "LNP/AAV penetration through mucus is a major translational barrier."
            )

        return warnings

    def format_context(self, technology: str, tissue: str,
                        mouse_efficiency_pct: float = None) -> str:
        """Format translation intelligence as text for LLM context injection."""
        pred = self.predict_human_efficiency(technology, tissue, mouse_efficiency_pct)

        lines = ["### 🐭→🧑 Cross-Species Translation Intelligence"]
        lines.append(f"**Technology:** {technology} | **Tissue:** {tissue}")
        lines.append(f"**Translation Quality:** {pred['quality']}")

        if pred.get("tissue_avg_ratio"):
            lines.append(f"**Average mouse→human ratio:** {pred['tissue_avg_ratio']:.2f} "
                         f"(range: {pred.get('tissue_range', 'N/A')})")

        if pred.get("predicted_human_efficiency_pct"):
            lines.append(f"**Predicted human efficiency:** {pred['predicted_human_efficiency_pct']}%")
            lines.append(f"  Basis: {pred['prediction_basis']}")

        if pred.get("note"):
            lines.append(f"**Note:** {pred['note']}")

        for m in pred.get("specific_matches", [])[:2]:
            lines.append(f"\n**Reference case ({m.get('target', 'N/A')}):**")
            lines.append(f"- Mouse: {m.get('mouse_efficiency', 'N/A')}")
            lines.append(f"- Human: {m.get('human_efficiency', 'N/A')}")
            if m.get("ref"):
                lines.append(f"- Ref: {m['ref']}")

        # Critical warnings
        warnings = self.get_critical_warnings(tissue)
        for w in warnings:
            lines.append(f"\n{w}")

        return "\n".join(lines) if len(lines) > 2 else ""

    @staticmethod
    def query_needs_translation(query: str) -> bool:
        """Heuristic: does this query involve mouse→human translation?"""
        q = query.lower()
        triggers = [
            r'mouse|小鼠|murine|动物模型|animal model',
            r'translat|转化|预测.*人|predict.*human',
            r'preclinical|临床前|in vivo.*efficien',
            r'小鼠.*效率|mouse.*efficien|efficiency.*human',
            r'能否.*外推|extrapolat',
        ]
        return any(re.search(pat, q, re.I) for pat in triggers)
