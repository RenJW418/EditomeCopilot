"""
Delivery Decision Tree
======================
Parameterized rule engine for selecting optimal delivery methods
for gene editing therapeutics.

Pipeline:  Target Tissue → Editing Technology → Payload Size →
           Repeat Dosing Need → Immune Status → Manufacturing Scale
           → Ranked Delivery Recommendations

Data: Literature-curated delivery method characteristics, clinical precedents.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Delivery Methods — Comprehensive Catalog
# ─────────────────────────────────────────────────────────────────────────────
DELIVERY_METHODS: Dict[str, Dict[str, Any]] = {
    "LNP_IV": {
        "name": "Lipid Nanoparticle (IV)",
        "full_name": "Ionizable LNP — Intravenous",
        "payload_types": ["mRNA", "sgRNA", "RNP (limited)"],
        "max_payload_kb": 10.0,
        "target_tissues": ["liver (primary)", "spleen (secondary)"],
        "tropism": "Liver-tropic (~90% hepatic accumulation)",
        "redosable": True,
        "immunogenicity": "LOW (no anti-vector immunity; PEG antibodies possible with repeat dosing)",
        "manufacturing_scale": "LARGE (industrial scalable, COVID vaccine precedent)",
        "clinical_precedent": "NTLA-2001 (TTR KO), VERVE-101 (PCSK9 ABE) — both Phase I/Ib",
        "pros": [
            "Transient expression (mRNA degrades in ~48h → no persistent Cas9)",
            "Redosable (no capsid immunity)",
            "Industrially scalable (mRNA/LNP platform)",
            "Liver editing highly efficient (0.1-1.0 mg/kg dose range)",
        ],
        "cons": [
            "Predominantly liver-tropic (non-liver tissues very low uptake)",
            "Dose-dependent hepatotoxicity (ALT/AST elevation)",
            "PEG antibodies may reduce efficacy with repeat dosing",
            "Cold chain required (-20°C)",
        ],
        "cost_estimate": "$50K-200K per treatment",
        "refs": ["10.1056/NEJMoa2107454", "10.1056/NEJMoa2303223"],
    },
    "LNP_intrathecal": {
        "name": "LNP (Intrathecal)",
        "full_name": "Ionizable LNP — Intrathecal Injection",
        "payload_types": ["mRNA", "sgRNA"],
        "max_payload_kb": 10.0,
        "target_tissues": ["spinal cord neurons", "brain (limited)"],
        "tropism": "CNS (spinal cord > brain stem > cortex)",
        "redosable": True,
        "immunogenicity": "LOW",
        "manufacturing_scale": "LARGE",
        "clinical_precedent": "Preclinical only (NHP studies by Intellia, Arcturus)",
        "pros": ["Bypass BBB", "Redosable", "Transient expression"],
        "cons": ["Invasive (lumbar puncture)", "Limited brain penetration from spinal route", "Distribution gradient"],
        "cost_estimate": "$100K-300K per treatment",
        "refs": [],
    },
    "LNP_nebulized": {
        "name": "LNP (Nebulized)",
        "full_name": "Ionizable LNP — Inhaled/Nebulized",
        "payload_types": ["mRNA", "sgRNA"],
        "max_payload_kb": 10.0,
        "target_tissues": ["lung epithelium", "conducting airways"],
        "tropism": "Lung airway epithelium (apical surface)",
        "redosable": True,
        "immunogenicity": "LOW",
        "manufacturing_scale": "MEDIUM-LARGE",
        "clinical_precedent": "Preclinical (CF models in mice/NHP)",
        "pros": ["Non-invasive", "Direct lung targeting", "Redosable", "Patient self-administrable"],
        "cons": ["Mucus barrier (especially in CF)", "Uneven distribution", "Early-stage technology"],
        "cost_estimate": "$50K-150K per treatment",
        "refs": ["10.1038/s41587-023-02004-y"],
    },
    "AAV_systemic": {
        "name": "AAV (Systemic IV)",
        "full_name": "Adeno-Associated Virus — Intravenous",
        "payload_types": ["DNA (transgene)", "SaCas9 + sgRNA (within 4.7kb)"],
        "max_payload_kb": 4.7,
        "target_tissues": ["liver (AAV8/LK03)", "muscle (AAV9/MyoAAV)", "CNS (AAV9/AAV.PHP.eB*)"],
        "tropism": "Serotype-dependent; broad (AAV9) or liver-tropic (AAV8)",
        "redosable": False,
        "immunogenicity": "HIGH (anti-capsid antibodies; 30-60% pre-existing immunity)",
        "manufacturing_scale": "SMALL-MEDIUM (bioreactor-dependent, yield challenges)",
        "clinical_precedent": "EDIT-101 (AAV5, retinal), SRP-9001 (AAVrh74, muscle), Zolgensma (AAV9, CNS)",
        "pros": [
            "Broad tissue tropism (serotype engineering)",
            "Long-term expression in non-dividing cells",
            "Delivery to difficult tissues (muscle, CNS, retina)",
        ],
        "cons": [
            "Size limit 4.7kb (SpCas9 alone is 4.2kb → very tight)",
            "NOT redosable (anti-capsid immunity)",
            "30-60% patients excluded (pre-existing antibodies)",
            "High-dose liver toxicity (hepatic SAEs reported at >1e14 vg/kg)",
            "Integration risk (low but non-zero)",
            "Extremely expensive manufacturing ($500K-2M/dose at scale)",
        ],
        "cost_estimate": "$500K-2.0M per treatment",
        "refs": ["10.1056/NEJMoa2309915"],
    },
    "AAV_subretinal": {
        "name": "AAV (Subretinal)",
        "full_name": "Adeno-Associated Virus — Subretinal Injection",
        "payload_types": ["DNA", "SaCas9 + sgRNA"],
        "max_payload_kb": 4.7,
        "target_tissues": ["photoreceptors", "RPE"],
        "tropism": "Photoreceptor/RPE (AAV2/5/8/9 variants)",
        "redosable": True,  # Contralateral eye possible
        "immunogenicity": "LOW (immune-privileged site)",
        "manufacturing_scale": "SMALL",
        "clinical_precedent": "Luxturna (AAV2, RPE65), EDIT-101 (AAV5, CEP290)",
        "pros": ["Immune-privileged site", "Small target tissue → lower dose", "Luxturna precedent"],
        "cons": ["Invasive surgery", "Only covers injection bleb area", "Retinal detachment risk"],
        "cost_estimate": "$400K-850K per eye",
        "refs": ["10.1056/NEJMoa2309915"],
    },
    "RNP_electroporation": {
        "name": "RNP Electroporation (Ex Vivo)",
        "full_name": "Ribonucleoprotein — Electroporation",
        "payload_types": ["Cas9/Cas12a protein + sgRNA (pre-complexed)"],
        "max_payload_kb": None,  # Protein delivery, no kb limit
        "target_tissues": ["HSC (ex vivo)", "T cells (ex vivo)", "iPSC (ex vivo)"],
        "tropism": "N/A (ex vivo cell delivery)",
        "redosable": True,  # Can re-electroporate cells
        "immunogenicity": "N/A (ex vivo — protein degrades before transplant)",
        "manufacturing_scale": "SMALL (per-patient manufacturing)",
        "clinical_precedent": "Casgevy (HSC), CTX110 (T cells), numerous CAR-T trials",
        "pros": [
            "Highest editing efficiency (>90%)",
            "No payload size limit",
            "Transient — Cas9 protein degrades in 24-48h",
            "No vector immunity concerns",
            "Well-established GMP processes",
        ],
        "cons": [
            "Ex vivo only (requires cell collection → editing → reinfusion)",
            "Myeloablative conditioning required for HSC engraftment",
            "Per-patient manufacturing (autologous is $1-2M)",
            "Cell viability loss (10-20% from electroporation)",
        ],
        "cost_estimate": "$500K-2.1M per treatment (autologous)",
        "refs": ["10.1056/NEJMoa2031054"],
    },
    "mRNA_electroporation": {
        "name": "mRNA Electroporation (Ex Vivo)",
        "full_name": "Editor mRNA + sgRNA — Electroporation",
        "payload_types": ["mRNA (any editor)", "sgRNA", "pegRNA"],
        "max_payload_kb": 15.0,
        "target_tissues": ["HSC (ex vivo)", "T cells (ex vivo)"],
        "tropism": "N/A (ex vivo)",
        "redosable": True,
        "immunogenicity": "N/A",
        "manufacturing_scale": "MEDIUM (mRNA GMP established)",
        "clinical_precedent": "Multiple base editing + prime editing ex vivo trials",
        "pros": [
            "Can deliver large editors (ABE, CBE, PE — all >5kb mRNA)",
            "Good editing efficiency (60-80%)",
            "Transient expression",
        ],
        "cons": [
            "Ex vivo only",
            "Slightly lower efficiency than RNP for small editors",
            "mRNA quality critical (cap, polyA, modified nucleotides)",
        ],
        "cost_estimate": "$500K-2.0M per treatment",
        "refs": [],
    },
    "VLP": {
        "name": "Virus-Like Particle",
        "full_name": "eVLP / Base Editing VLP",
        "payload_types": ["Cas9/BE/PE protein (packaged in lentiviral-derived particle)"],
        "max_payload_kb": None,
        "target_tissues": ["liver", "HSC", "brain (with engineered envelope)"],
        "tropism": "Pseudotypable (VSV-G for broad, engineered for tissue-specific)",
        "redosable": True,
        "immunogenicity": "MEDIUM (envelope protein immunity possible)",
        "manufacturing_scale": "SMALL-MEDIUM (early-stage production)",
        "clinical_precedent": "Preclinical only (Beam Therapeutics, David Liu lab)",
        "pros": [
            "Delivers protein (transient, no DNA integration risk)",
            "No payload size limit",
            "Potentially tissue-retargetable",
            "Redosable (pseudotyping evades immunity)",
        ],
        "cons": [
            "Early-stage manufacturing",
            "Lower efficiency than RNP ex vivo",
            "Complex production (requires 3-4 plasmid transfection)",
        ],
        "cost_estimate": "Unknown (preclinical)",
        "refs": ["10.1038/s41587-023-01779-2"],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Decision Rules: Tissue → Delivery Scoring
# ─────────────────────────────────────────────────────────────────────────────
TISSUE_DELIVERY_SCORES: Dict[str, Dict[str, int]] = {
    # Scores 0-10: 10 = ideal, 0 = not applicable
    "liver": {
        "LNP_IV": 10, "AAV_systemic": 7, "VLP": 5,
        "LNP_intrathecal": 0, "LNP_nebulized": 0, "AAV_subretinal": 0,
        "RNP_electroporation": 0, "mRNA_electroporation": 0,
    },
    "HSC": {
        "RNP_electroporation": 10, "mRNA_electroporation": 9,
        "VLP": 4, "LNP_IV": 0, "AAV_systemic": 0,
        "LNP_intrathecal": 0, "LNP_nebulized": 0, "AAV_subretinal": 0,
    },
    "T cells": {
        "RNP_electroporation": 10, "mRNA_electroporation": 9,
        "VLP": 3, "LNP_IV": 0, "AAV_systemic": 0,
        "LNP_intrathecal": 0, "LNP_nebulized": 0, "AAV_subretinal": 0,
    },
    "muscle": {
        "AAV_systemic": 7, "LNP_IV": 2, "VLP": 3,
        "RNP_electroporation": 0, "mRNA_electroporation": 0,
        "LNP_intrathecal": 0, "LNP_nebulized": 0, "AAV_subretinal": 0,
    },
    "retina": {
        "AAV_subretinal": 10, "LNP_IV": 1, "VLP": 3, "AAV_systemic": 2,
        "RNP_electroporation": 0, "mRNA_electroporation": 0,
        "LNP_intrathecal": 0, "LNP_nebulized": 0,
    },
    "brain": {
        "AAV_systemic": 5, "LNP_intrathecal": 7, "VLP": 4,
        "LNP_IV": 1, "LNP_nebulized": 0, "AAV_subretinal": 0,
        "RNP_electroporation": 0, "mRNA_electroporation": 0,
    },
    "lung": {
        "LNP_nebulized": 8, "AAV_systemic": 4, "LNP_IV": 2, "VLP": 3,
        "RNP_electroporation": 0, "mRNA_electroporation": 0,
        "LNP_intrathecal": 0, "AAV_subretinal": 0,
    },
}

# Editor → Payload size + delivery compatibility
EDITOR_PAYLOAD: Dict[str, Dict[str, Any]] = {
    "SpCas9": {"size_kb": 4.2, "compatible_delivery": ["LNP_IV", "AAV_systemic", "RNP_electroporation", "mRNA_electroporation", "VLP"]},
    "SaCas9": {"size_kb": 3.2, "compatible_delivery": ["LNP_IV", "AAV_systemic", "RNP_electroporation", "mRNA_electroporation", "VLP"]},
    "Cas12a": {"size_kb": 3.9, "compatible_delivery": ["LNP_IV", "AAV_systemic", "RNP_electroporation", "mRNA_electroporation", "VLP"]},
    "ABE8e": {"size_kb": 5.4, "compatible_delivery": ["LNP_IV", "mRNA_electroporation", "VLP"]},  # Too large for single AAV
    "CBE4": {"size_kb": 5.6, "compatible_delivery": ["LNP_IV", "mRNA_electroporation", "VLP"]},
    "PE2": {"size_kb": 6.3, "compatible_delivery": ["LNP_IV", "mRNA_electroporation", "VLP"]},  # Requires dual-AAV split
    "PE5max": {"size_kb": 6.5, "compatible_delivery": ["LNP_IV", "mRNA_electroporation", "VLP"]},
    "CRISPRi": {"size_kb": 5.0, "compatible_delivery": ["LNP_IV", "AAV_systemic", "mRNA_electroporation"]},
    "Cas13": {"size_kb": 4.5, "compatible_delivery": ["LNP_IV", "AAV_systemic", "mRNA_electroporation"]},
}


class DeliveryDecisionTree:
    """Parameterized decision engine for editing delivery method selection."""

    def __init__(self):
        self.methods = DELIVERY_METHODS
        self.tissue_scores = TISSUE_DELIVERY_SCORES
        self.editor_payload = EDITOR_PAYLOAD

    def recommend(self, tissue: str, editor: str = None,
                  needs_redosing: bool = False,
                  exclude_high_cost: bool = False,
                  clinical_only: bool = False) -> List[Dict[str, Any]]:
        """
        Score and rank delivery methods for given parameters.

        Args:
            tissue: Target tissue (liver, HSC, T cells, muscle, retina, brain, lung)
            editor: Optional editor name (SpCas9, ABE8e, PE2, etc.)
            needs_redosing: Whether repeat dosing is required
            exclude_high_cost: Exclude methods > $1M
            clinical_only: Only include methods with clinical precedent

        Returns:
            Ranked list of delivery recommendations with scores and rationale
        """
        tissue_lower = tissue.lower()

        # Find best matching tissue key
        tissue_key = None
        for k in self.tissue_scores:
            if k.lower() in tissue_lower or tissue_lower in k.lower():
                tissue_key = k
                break
        if not tissue_key:
            # Fallback: try partial match
            for k in self.tissue_scores:
                if any(t in k.lower() for t in tissue_lower.split()):
                    tissue_key = k
                    break

        if not tissue_key:
            return [{"method": "UNKNOWN", "score": 0,
                     "note": f"No delivery data for tissue '{tissue}'. Consider LNP (if liver-proximate) or AAV (broad)."}]

        scores = self.tissue_scores[tissue_key]

        # Build candidate list
        candidates = []
        for method_id, base_score in scores.items():
            if base_score == 0:
                continue

            method_info = self.methods.get(method_id, {})
            score = base_score

            # Editor compatibility check
            if editor:
                editor_info = None
                for ename, einfo in self.editor_payload.items():
                    if ename.lower() in editor.lower() or editor.lower() in ename.lower():
                        editor_info = einfo
                        break

                if editor_info:
                    if method_id not in editor_info.get("compatible_delivery", []):
                        score -= 5  # Heavy penalty
                        if score < 0:
                            continue
                    # Size check for AAV
                    if "AAV" in method_id and editor_info.get("size_kb", 0) > 4.7:
                        score -= 3  # Needs split-intein approach

            # Redosing requirement
            if needs_redosing and not method_info.get("redosable", False):
                score -= 4

            # Cost filter
            if exclude_high_cost:
                cost = method_info.get("cost_estimate", "")
                if "2.0M" in cost or "2.1M" in cost:
                    score -= 3

            # Clinical precedent bonus
            precedent = method_info.get("clinical_precedent", "")
            if "Phase" in precedent or "FDA" in precedent or "approved" in precedent.lower():
                score += 2
            if clinical_only and "Preclinical" in precedent:
                continue

            candidates.append({
                "method_id": method_id,
                "name": method_info.get("name", method_id),
                "score": score,
                "tropism": method_info.get("tropism", ""),
                "redosable": method_info.get("redosable", False),
                "max_payload_kb": method_info.get("max_payload_kb"),
                "immunogenicity": method_info.get("immunogenicity", ""),
                "clinical_precedent": method_info.get("clinical_precedent", ""),
                "pros": method_info.get("pros", []),
                "cons": method_info.get("cons", []),
                "cost_estimate": method_info.get("cost_estimate", ""),
            })

        # Sort by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates

    def format_context(self, tissue: str, editor: str = None,
                        needs_redosing: bool = False) -> str:
        """Format delivery recommendations as structured text for LLM."""
        recs = self.recommend(tissue, editor, needs_redosing)

        if not recs:
            return ""

        lines = ["### 🚀 Delivery Method Decision Engine"]
        lines.append(f"**Target tissue:** {tissue}")
        if editor:
            lines.append(f"**Editor:** {editor}")

        for i, r in enumerate(recs[:3], 1):
            lines.append(f"\n**#{i} {r['name']}** (score: {r['score']}/12)")
            lines.append(f"- Tropism: {r['tropism']}")
            lines.append(f"- Redosable: {'Yes' if r['redosable'] else 'No'}")
            if r.get("max_payload_kb"):
                lines.append(f"- Max payload: {r['max_payload_kb']} kb")
            lines.append(f"- Immunogenicity: {r['immunogenicity']}")
            lines.append(f"- Clinical precedent: {r['clinical_precedent']}")
            lines.append(f"- Cost: {r['cost_estimate']}")

            if r.get("pros"):
                lines.append(f"- Pros: {'; '.join(r['pros'][:2])}")
            if r.get("cons"):
                lines.append(f"- Cons: {'; '.join(r['cons'][:2])}")

        return "\n".join(lines)

    @staticmethod
    def query_needs_delivery(query: str) -> bool:
        """Heuristic: does this query involve delivery method selection?"""
        q = query.lower()
        triggers = [
            r'deliver|递送|投递',
            r'lnp|aav|nanoparticle|纳米颗粒|脂质体',
            r'tropism|嗜性',
            r'in vivo.*how|怎么.*体内',
            r'载体|vector|carrier|vehicle',
            r'electropor|电穿孔',
            r'how to.*get.*into|如何.*导入',
        ]
        return any(re.search(pat, q, re.I) for pat in triggers)
