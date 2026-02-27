"""
Variant → Editing Strategy Resolver
=====================================
Given a genomic variant (ClinVar ID, HGVS notation, or gene+mutation),
automatically reason about the optimal editing strategy.

Pipeline:
1. Parse variant input (HGVS, ClinVar, gene+mutation text)
2. Classify mutation type (SNV, insertion, deletion, frameshift, etc.)
3. Determine PAM accessibility around the target locus
4. Evaluate editing window compatibility for each editor
5. Quantify bystander editing risk
6. Rank strategies and return evidence-linked recommendations
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Known variant → editing strategy mappings (literature-curated)
# ─────────────────────────────────────────────────────────────────────────────
CURATED_VARIANTS: Dict[str, Dict[str, Any]] = {
    # HBB E6V — Sickle Cell Disease 核心突变
    "HBB:c.20A>T": {
        "gene": "HBB", "protein": "E6V", "disease": "Sickle Cell Disease",
        "clinvar": "VCV000036610", "chromosome": "chr11:5227002",
        "ref": "A", "alt": "T", "mutation_type": "SNV",
        "strategies": [
            {
                "technology": "ABE (A→G on antisense)",
                "mechanism": "ABE targets antisense strand: T→C (complements A→G), reverting pathogenic T back to A",
                "pam_site": "NGA (xCas9) or NGG (SpCas9) — multiple PAM sites within ±20bp",
                "editing_window": "Target A at position 5 in protospacer — ideal for ABE8e (window 4-7)",
                "bystander_risk": "One additional A at position 7; 5-15% bystander editing (silent A→G)",
                "expected_efficiency": "58-68% in human HSCs (Newby et al. 2021 Nature)",
                "evidence_level": "Level 2 (in vivo mouse + ex vivo human HSC)",
                "pros": "No DSB; low off-target; direct mutation correction",
                "cons": "Bystander A at pos 7; delivery requires mRNA/RNP (too large for single AAV)",
                "key_ref": "10.1038/s41586-021-03609-w",
            },
            {
                "technology": "Prime Editing (PE5max)",
                "mechanism": "pegRNA encodes exact A>T→A reversion; RT writes correct sequence",
                "pam_site": "NGG — multiple sites available",
                "editing_window": "Flexible — PE can correct any base within ±40bp of nick site",
                "bystander_risk": "None (only specified edit is installed)",
                "expected_efficiency": "30-40% in HSCs (Everette et al. 2023 Nature)",
                "evidence_level": "Level 2 (in vivo mouse)",
                "pros": "Zero bystander; versatile; installs exact correction",
                "cons": "Lower efficiency than ABE; larger payload (~6.3kb mRNA)",
                "key_ref": "10.1038/s41586-023-06004-3",
            },
            {
                "technology": "CRISPR-Cas9 KO (BCL11A enhancer)",
                "mechanism": "Indirect: disrupt BCL11A erythroid enhancer → reactivate HbF → compensate for HbS",
                "pam_site": "Well-validated NGG site in BCL11A +58 enhancer",
                "editing_window": "N/A — produces indels at cut site",
                "bystander_risk": "Indels are heterogeneous; chromothripsis risk (very low)",
                "expected_efficiency": "95% HbF induction; FDA-approved (Casgevy)",
                "evidence_level": "Level 1 (FDA Approved, Phase III)",
                "pros": "Clinically validated; highest evidence level; proven safety/efficacy",
                "cons": "Does not correct underlying mutation; requires myeloablative conditioning",
                "key_ref": "10.1056/NEJMoa2031054",
            },
        ],
    },
    # PCSK9 — Familial Hypercholesterolemia
    "PCSK9:splice_disruption": {
        "gene": "PCSK9", "protein": "Splice site disruption", "disease": "Familial Hypercholesterolemia",
        "clinvar": "N/A (therapeutic KO)", "chromosome": "chr1:55505221",
        "ref": "multiple", "alt": "multiple", "mutation_type": "therapeutic_KO",
        "strategies": [
            {
                "technology": "ABE (splice site disruption)",
                "mechanism": "ABE edits splice donor of PCSK9 exon → aberrant splicing → loss of function",
                "pam_site": "NGG at PCSK9 exon 1 splice donor",
                "editing_window": "Splice donor A at optimal position in ABE window",
                "bystander_risk": "Minimal — only target A in functional position",
                "expected_efficiency": "~60% base conversion in NHP liver; 55% LDL reduction in human Phase Ib",
                "evidence_level": "Level 1 (Phase Ib — VERVE-101)",
                "pros": "One-time treatment; no DSB; LNP delivery proven in human",
                "cons": "Irreversible; liver-specific (LNP tropism); long-term follow-up pending",
                "key_ref": "10.1056/NEJMoa2303223",
            },
        ],
    },
    # TTR — ATTR Amyloidosis
    "TTR:KO": {
        "gene": "TTR", "protein": "TTR knockout", "disease": "ATTR Amyloidosis",
        "clinvar": "N/A (therapeutic KO)", "chromosome": "chr18:31592900",
        "ref": "N/A", "alt": "N/A", "mutation_type": "therapeutic_KO",
        "strategies": [
            {
                "technology": "CRISPR-Cas9 KO (in vivo LNP)",
                "mechanism": "SpCas9 induces DSB in TTR exon → NHEJ → frameshift → loss of hepatic TTR production",
                "pam_site": "NGG in TTR exon 2",
                "editing_window": "N/A (nuclease cut)",
                "bystander_risk": "Indels heterogeneous; no bystander per se",
                "expected_efficiency": "87% TTR serum reduction (single dose, Phase I)",
                "evidence_level": "Level 1 (Phase I — NTLA-2001)",
                "pros": "First in vivo CRISPR in human; one-time treatment; 87% knockdown",
                "cons": "DSB-based (rare chromosomal abnormality risk); irreversible",
                "key_ref": "10.1056/NEJMoa2107454",
            },
        ],
    },
    # DMD exon 51 skipping
    "DMD:exon51_skip": {
        "gene": "DMD", "protein": "Exon 51 skipping → truncated dystrophin",
        "disease": "Duchenne Muscular Dystrophy",
        "clinvar": "N/A", "chromosome": "chrX:31496382",
        "ref": "N/A", "alt": "N/A", "mutation_type": "exon_skipping",
        "strategies": [
            {
                "technology": "CRISPR-Cas9 (dual-guide exon deletion)",
                "mechanism": "Two guides flanking exon 51 → delete exon → restore reading frame → truncated but functional dystrophin",
                "pam_site": "Intronic NGG sites flanking exon 51",
                "editing_window": "N/A (large deletion between two cut sites)",
                "bystander_risk": "Large deletion heterogeneity; potential inversions",
                "expected_efficiency": "5-50% dystrophin restoration in mdx mice",
                "evidence_level": "Level 2 (animal models)",
                "pros": "Permanent correction; addresses ~14% of DMD patients",
                "cons": "Dual-guide complexity; large deletion heterogeneity; requires systemic AAV9",
                "key_ref": "10.1126/science.aad5143",
            },
            {
                "technology": "Base Editing (splice site modification)",
                "mechanism": "CBE/ABE edits splice donor/acceptor → exon skipping without DSB",
                "pam_site": "Splice site-adjacent PAM required",
                "editing_window": "Splice consensus position must fall in editing window",
                "bystander_risk": "Other C/A in window may be edited",
                "expected_efficiency": "30-70% exon skipping in cell models",
                "evidence_level": "Level 2-3 (in vitro / animal)",
                "pros": "No DSB; single guide; more predictable outcome",
                "cons": "Requires favorable splice site position relative to PAM/editing window",
                "key_ref": "10.1038/s41586-020-2950-4",
            },
        ],
    },
    # SERPINA1 Z allele — Alpha-1 Antitrypsin Deficiency
    "SERPINA1:E342K": {
        "gene": "SERPINA1", "protein": "E342K (Z allele)", "disease": "Alpha-1 Antitrypsin Deficiency",
        "clinvar": "VCV000018981", "chromosome": "chr14:94378610",
        "ref": "G", "alt": "A", "mutation_type": "SNV",
        "strategies": [
            {
                "technology": "RNA Editing (endogenous ADAR recruitment)",
                "mechanism": "Stereopure ASO recruits endogenous ADAR to correct A→I(G) at the Z allele mRNA",
                "pam_site": "N/A (guide RNA directed)",
                "editing_window": "Target A in dsRNA structure formed by ASO hybridization",
                "bystander_risk": "Transcriptome-wide A-to-I: 0.01-1% (dose-dependent)",
                "expected_efficiency": "~40% editing at target site (Phase I/II — WVE-006)",
                "evidence_level": "Level 1 (Phase I/II)",
                "pros": "Reversible; no DNA modification; redosable; small payload (ASO only)",
                "cons": "Requires repeated dosing; transcriptome-wide off-target A-to-I",
                "key_ref": "Wave Life Sciences WVE-006 clinical data",
            },
        ],
    },
    # CEP290 IVS26 — LCA10
    "CEP290:IVS26": {
        "gene": "CEP290", "protein": "IVS26 intronic mutation", "disease": "Leber Congenital Amaurosis 10",
        "clinvar": "VCV000001347", "chromosome": "chr12:88071099",
        "ref": "A", "alt": "G", "mutation_type": "intronic_SNV",
        "strategies": [
            {
                "technology": "CRISPR-Cas9 (dual-guide, intronic deletion)",
                "mechanism": "Two guides flanking the aberrant splice site → delete intronic mutation → restore normal splicing",
                "pam_site": "Intronic NGG sites flanking IVS26 mutation",
                "editing_window": "N/A (deletion between cuts)",
                "bystander_risk": "Deletion size heterogeneity",
                "expected_efficiency": "Improved visual acuity in Phase I/II (EDIT-101)",
                "evidence_level": "Level 1 (Phase I/II)",
                "pros": "Subretinal AAV5 delivery (small target tissue); proven photoreceptor transduction",
                "cons": "Requires invasive subretinal injection; limited to LCA10",
                "key_ref": "10.1056/NEJMoa2309915",
            },
        ],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Generic Mutation Type → Strategy Rules
# ─────────────────────────────────────────────────────────────────────────────
MUTATION_TYPE_RULES: Dict[str, List[Dict[str, str]]] = {
    "SNV": [
        {"technology": "ABE (A→G / T→C)", "applicability": "If target is A→G or T→C (antisense)", "priority": 1,
         "rationale": "No DSB, high efficiency (50-70%), low off-target. First choice for compatible transitions."},
        {"technology": "CBE (C→T / G→A)", "applicability": "If target is C→T or G→A (antisense)", "priority": 1,
         "rationale": "No DSB, 60-80% efficiency. Watch for bystander C editing in window."},
        {"technology": "Prime Editing", "applicability": "Any SNV (including transversions)", "priority": 2,
         "rationale": "Handles all 12 base substitutions. No bystander. Lower efficiency (30-60%)."},
        {"technology": "CRISPR-Cas9 + HDR", "applicability": "Any SNV with available donor template", "priority": 3,
         "rationale": "Requires DSB + donor. Low efficiency in non-dividing cells. NHEJ competes."},
    ],
    "small_insertion": [
        {"technology": "Prime Editing", "applicability": "Insertions up to ~40bp", "priority": 1,
         "rationale": "pegRNA encodes insertion template. No DSB. Precise. PE5max preferred."},
        {"technology": "CRISPR-Cas9 + HDR", "applicability": "Larger insertions with donor template", "priority": 2,
         "rationale": "Can insert larger fragments but requires DSB + homologous donor."},
    ],
    "small_deletion": [
        {"technology": "Prime Editing", "applicability": "Deletions up to ~80bp", "priority": 1,
         "rationale": "Precise deletion specified by pegRNA. No bystander."},
        {"technology": "CRISPR-Cas9 (dual-guide)", "applicability": "Larger deletions (exon removal)", "priority": 2,
         "rationale": "Two guides flanking region. Deletion size heterogeneous."},
    ],
    "frameshift": [
        {"technology": "CRISPR-Cas9 (exon skipping)", "applicability": "Out-of-frame exons (DMD model)", "priority": 1,
         "rationale": "Disrupt splice sites to skip mutant exon → restore reading frame."},
        {"technology": "Base Editing (splice site)", "applicability": "Splice donor/acceptor editing", "priority": 2,
         "rationale": "Modify splice consensus → exon skipping without DSB."},
        {"technology": "Prime Editing", "applicability": "Small frameshift correction", "priority": 2,
         "rationale": "Insert/delete 1-2bp to restore frame. Precise but lower efficiency."},
    ],
    "repeat_expansion": [
        {"technology": "CRISPR-Cas9 (dual-guide excision)", "applicability": "Excise expanded repeat", "priority": 1,
         "rationale": "Two guides flanking repeat → delete. Used in HTT/SCA models."},
        {"technology": "CRISPRi (transcriptional silencing)", "applicability": "Silence expanded allele", "priority": 2,
         "rationale": "dCas9-KRAB silences mutant allele. Reversible. No DNA modification."},
    ],
    "splicing_defect": [
        {"technology": "ABE/CBE (splice site correction)", "applicability": "Splice donor/acceptor mutation", "priority": 1,
         "rationale": "Direct correction of splice consensus sequence. High efficiency."},
        {"technology": "RNA Editing (ADAR)", "applicability": "Transient correction at mRNA level", "priority": 2,
         "rationale": "Correct mRNA without touching DNA. Reversible. Requires repeated dosing."},
        {"technology": "CRISPR-Cas9 (intronic deletion)", "applicability": "Deep intronic mutations", "priority": 2,
         "rationale": "Delete aberrant splice site (CEP290 IVS26 model)."},
    ],
    "gain_of_function": [
        {"technology": "CRISPR-Cas9 KO (allele-specific)", "applicability": "Silence gain-of-function allele", "priority": 1,
         "rationale": "Disrupt mutant allele while preserving wild-type. Requires allele-specific PAM/guide."},
        {"technology": "CRISPRi", "applicability": "Transcriptional repression of mutant allele", "priority": 2,
         "rationale": "Reversible silencing. Useful when permanent KO too risky."},
        {"technology": "Cas13 (RNA knockdown)", "applicability": "Degrade mutant mRNA", "priority": 2,
         "rationale": "Allele-specific RNA targeting. Transient. No DNA modification."},
    ],
    "therapeutic_KO": [
        {"technology": "CRISPR-Cas9 KO", "applicability": "Loss-of-function is therapeutic", "priority": 1,
         "rationale": "Most mature; highest clinical evidence (NTLA-2001, Casgevy)."},
        {"technology": "ABE (splice disruption)", "applicability": "KO via splice site editing", "priority": 1,
         "rationale": "No DSB. Used in VERVE-101 (PCSK9). Preferred for in vivo liver."},
        {"technology": "CRISPRi", "applicability": "Reversible knockdown", "priority": 2,
         "rationale": "If permanent KO too risky. Requires sustained expression."},
    ],
}

# Nucleotide change → compatible base editor
_BASE_EDITOR_MAP = {
    ("A", "G"): "ABE",  ("T", "C"): "ABE (antisense)",
    ("C", "T"): "CBE",  ("G", "A"): "CBE (antisense)",
    # Transversions → PE only
    ("A", "T"): "PE", ("A", "C"): "PE", ("T", "A"): "PE", ("T", "G"): "PE",
    ("C", "A"): "PE", ("C", "G"): "PE", ("G", "T"): "PE", ("G", "C"): "PE",
}


class VariantResolver:
    """Resolve genomic variants to optimal editing strategies."""

    def __init__(self, llm_client=None, almanac=None):
        self.llm = llm_client
        self.almanac = almanac
        self.curated = CURATED_VARIANTS
        self.rules = MUTATION_TYPE_RULES

    # ─── Parse Variant Input ──────────────────────────────────────────────

    def parse_variant(self, query: str) -> Dict[str, Any]:
        """
        Extract variant information from natural language or structured notation.

        Supports:
          - HGVS: NM_000518.5:c.20A>T or c.20A>T
          - Gene+protein: HBB E6V, HBB p.Glu6Val
          - Gene+mutation: PCSK9 knockout, TTR disruption
          - ClinVar: VCV000036610
          - Natural language: "sickle cell mutation", "PCSK9 loss of function"
        """
        q = query.strip()
        result = {"raw_input": q, "gene": None, "mutation": None,
                  "mutation_type": None, "ref": None, "alt": None,
                  "hgvs": None, "clinvar_id": None}

        # HGVS coding variant
        hgvs_match = re.search(r'(NM_\d+\.\d+:)?c\.(\d+)([ACGT])>([ACGT])', q, re.I)
        if hgvs_match:
            result["hgvs"] = hgvs_match.group(0)
            result["ref"] = hgvs_match.group(3).upper()
            result["alt"] = hgvs_match.group(4).upper()
            result["mutation_type"] = "SNV"
            result["position"] = int(hgvs_match.group(2))

        # ClinVar ID
        cv_match = re.search(r'VCV\d{6,}', q)
        if cv_match:
            result["clinvar_id"] = cv_match.group(0)

        # Gene symbol
        gene_match = re.search(r'\b(HBB|PCSK9|TTR|DMD|CEP290|SERPINA1|BCL11A|KLKB1|HTT|CFTR|F8|F9|SMN1|SMN2|TRAC|B2M|CD19|HPK1|CCR5|G6PC1|CDKL5)\b', q, re.I)
        if gene_match:
            result["gene"] = gene_match.group(1).upper()

        # Protein-level single-letter
        prot_match = re.search(r'([A-Z])(\d+)([A-Z])\b', q)
        if prot_match and result["gene"]:
            result["mutation"] = prot_match.group(0)

        # Three-letter protein
        prot3_match = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})', q)
        if prot3_match:
            result["mutation"] = prot3_match.group(0)

        # Mutation type keywords
        q_lower = q.lower()
        type_keywords = {
            "knockout": "therapeutic_KO", "ko ": "therapeutic_KO", "disruption": "therapeutic_KO",
            "loss of function": "therapeutic_KO", "功能缺失": "therapeutic_KO",
            "insertion": "small_insertion", "ins": "small_insertion",
            "deletion": "small_deletion", "del": "small_deletion",
            "frameshift": "frameshift", "移码": "frameshift",
            "exon skip": "frameshift", "外显子跳跃": "frameshift",
            "repeat expansion": "repeat_expansion", "重复扩增": "repeat_expansion",
            "splice": "splicing_defect", "剪接": "splicing_defect",
            "gain of function": "gain_of_function", "功能获得": "gain_of_function",
            "snv": "SNV", "point mutation": "SNV", "点突变": "SNV",
        }
        if not result["mutation_type"]:
            for kw, mtype in type_keywords.items():
                if kw in q_lower:
                    result["mutation_type"] = mtype
                    break

        # Default to SNV if ref/alt detected
        if not result["mutation_type"] and result["ref"] and result["alt"]:
            result["mutation_type"] = "SNV"

        return result

    # ─── Lookup Curated Variants ──────────────────────────────────────────

    def lookup_curated(self, variant_info: Dict) -> Optional[Dict]:
        """Check if this variant has a curated entry."""
        gene = variant_info.get("gene", "")
        mutation = variant_info.get("mutation", "")
        q_lower = variant_info.get("raw_input", "").lower()

        for key, entry in self.curated.items():
            # Direct gene match
            if gene and gene.upper() == entry["gene"]:
                # Check if mutation matches or query is about this gene's main disease
                if mutation and mutation in key:
                    return entry
                if entry["disease"].lower() in q_lower:
                    return entry
                # Generic query about the gene
                if gene.lower() in q_lower and len(q_lower.split()) <= 5:
                    return entry

        # Disease-name match
        disease_map = {
            "sickle cell": "HBB:c.20A>T", "镰刀": "HBB:c.20A>T",
            "thalassemia": "HBB:c.20A>T", "地中海贫血": "HBB:c.20A>T",
            "hypercholesterolemia": "PCSK9:splice_disruption", "高胆固醇": "PCSK9:splice_disruption",
            "amyloidosis": "TTR:KO", "淀粉样变": "TTR:KO",
            "duchenne": "DMD:exon51_skip", "dmd": "DMD:exon51_skip",
            "lca": "CEP290:IVS26", "leber": "CEP290:IVS26",
            "alpha-1": "SERPINA1:E342K", "antitrypsin": "SERPINA1:E342K",
        }
        for kw, curated_key in disease_map.items():
            if kw in q_lower:
                return self.curated.get(curated_key)

        return None

    # ─── Rule-Based Strategy Generation ───────────────────────────────────

    def generate_strategies_by_rules(self, variant_info: Dict) -> List[Dict]:
        """Generate strategies from mutation type rules."""
        mtype = variant_info.get("mutation_type")
        if not mtype or mtype not in self.rules:
            return []

        strategies = []
        for rule in self.rules[mtype]:
            strategy = dict(rule)
            # If SNV with known ref/alt, add base editor compatibility
            if mtype == "SNV" and variant_info.get("ref") and variant_info.get("alt"):
                ref, alt = variant_info["ref"], variant_info["alt"]
                compatible_editor = _BASE_EDITOR_MAP.get((ref, alt), "PE")
                strategy["compatible_editor"] = compatible_editor
                if compatible_editor.startswith("ABE") and "ABE" in rule["technology"]:
                    strategy["priority"] = 0  # Boost compatible BE
                elif compatible_editor.startswith("CBE") and "CBE" in rule["technology"]:
                    strategy["priority"] = 0
            strategies.append(strategy)

        strategies.sort(key=lambda x: x.get("priority", 5))
        return strategies

    # ─── Full Resolution Pipeline ─────────────────────────────────────────

    def resolve(self, query: str) -> Dict[str, Any]:
        """
        Full pipeline: parse variant → lookup curated → apply rules → format.

        Returns structured result with strategies ranked by priority.
        """
        variant_info = self.parse_variant(query)
        result = {
            "variant_info": variant_info,
            "curated_entry": None,
            "rule_strategies": [],
            "recommendation": "",
        }

        # 1. Try curated database first
        curated = self.lookup_curated(variant_info)
        if curated:
            result["curated_entry"] = curated
            result["source"] = "curated_literature"

        # 2. Rule-based strategies
        rule_strategies = self.generate_strategies_by_rules(variant_info)
        result["rule_strategies"] = rule_strategies

        return result

    # ─── Format for LLM Context ──────────────────────────────────────────

    def format_context(self, resolution: Dict) -> str:
        """Format resolution result as structured text for LLM injection."""
        lines = ["### 🧬 Variant → Editing Strategy Resolver"]

        vi = resolution.get("variant_info", {})
        if vi.get("gene"):
            lines.append(f"**Gene:** {vi['gene']}  ")
        if vi.get("mutation"):
            lines.append(f"**Mutation:** {vi['mutation']}  ")
        if vi.get("mutation_type"):
            lines.append(f"**Type:** {vi['mutation_type']}  ")
        if vi.get("hgvs"):
            lines.append(f"**HGVS:** {vi['hgvs']}  ")
        if vi.get("ref") and vi.get("alt"):
            editor = _BASE_EDITOR_MAP.get((vi["ref"], vi["alt"]), "Prime Editing")
            lines.append(f"**Nucleotide change:** {vi['ref']}→{vi['alt']} → Compatible editor: **{editor}**  ")

        # Curated entry
        curated = resolution.get("curated_entry")
        if curated:
            lines.append(f"\n**Disease:** {curated['disease']}")
            lines.append(f"**Curated Editing Strategies (Literature-Validated):**")
            for i, s in enumerate(curated.get("strategies", []), 1):
                lines.append(f"\n**Strategy {i}: {s['technology']}**")
                lines.append(f"- Mechanism: {s['mechanism']}")
                lines.append(f"- PAM: {s.get('pam_site', 'N/A')}")
                lines.append(f"- Editing window: {s.get('editing_window', 'N/A')}")
                lines.append(f"- Bystander risk: {s.get('bystander_risk', 'N/A')}")
                lines.append(f"- Expected efficiency: {s.get('expected_efficiency', 'N/A')}")
                lines.append(f"- Evidence: {s.get('evidence_level', 'N/A')}")
                lines.append(f"- Pros: {s.get('pros', '')}")
                lines.append(f"- Cons: {s.get('cons', '')}")
                if s.get("key_ref"):
                    lines.append(f"- Key reference: doi:{s['key_ref']}")

        # Rule-based strategies
        rules = resolution.get("rule_strategies", [])
        if rules and not curated:
            lines.append(f"\n**Rule-Based Strategy Recommendations:**")
            for i, r in enumerate(rules, 1):
                lines.append(f"{i}. **{r['technology']}** (priority: {r.get('priority', '?')})")
                lines.append(f"   Applicability: {r['applicability']}")
                lines.append(f"   Rationale: {r['rationale']}")

        return "\n".join(lines) if len(lines) > 1 else ""

    # ─── Quick check: does query look like it needs variant resolution? ───

    @staticmethod
    def query_needs_resolution(query: str) -> bool:
        """Heuristic: does this query involve a specific variant or editing strategy design?"""
        q = query.lower()
        triggers = [
            r'c\.\d+[acgt]>[acgt]',           # HGVS
            r'vcv\d{6}',                        # ClinVar
            r'怎么.*编辑|如何.*纠正|how to.*edit|how to.*correct',
            r'editing strategy|编辑策略|编辑方案',
            r'什么.*最好.*治疗|best.*approach|which.*editor',
            r'point mutation|点突变|snv',
            r'knockout.*gene|基因.*敲除',
            r'纠正.*突变|correct.*mutation|fix.*mutation',
            r'[A-Z]{2,5}\s+[A-Z]\d+[A-Z]',    # Gene + protein mutation
        ]
        return any(re.search(pat, q, re.I) for pat in triggers)
