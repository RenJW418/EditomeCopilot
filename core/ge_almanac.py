"""
GEAlmanac — Gene Editing Almanac (Structured Knowledge Base)
============================================================
Inspired by: Molecular Oncology Almanac (MOAlmanac) in
  "A context-augmented LLM for accurate precision oncology medicine
   recommendations" (Jun et al. 2026, Cancer Cell)

Purpose: Provide structured, curated, data-driven facts that the LLM and
DecisionEngine can query directly — no hallucination possible for these fields.

Tables
------
1. Technology Registry     — Cas types, mechanisms, editing windows, PAM
2. Clinical Trials         — NCT ID, phase, disease, status, technology
3. Regulatory Approvals    — Drug name, date, indication, technology
4. Technology-Disease Matrix — Tech × Disease → evidence level, best efficiency
5. Safety Profiles         — Tech × context → off-target rate, immunogenicity
6. Guide RNA Database      — sgRNA, target gene, efficiency, off-target
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. Technology Registry
# ─────────────────────────────────────────────────────────────────────────────
TECHNOLOGY_REGISTRY: List[Dict[str, Any]] = [
    {
        "name": "CRISPR-Cas9 (SpCas9)",
        "category": "Nuclease (DSB)",
        "mechanism": "NHEJ / HDR via double-strand break",
        "pam": "NGG",
        "size_aa": 1368,
        "editing_window": "N/A (cut site: 3 bp upstream of PAM)",
        "max_efficiency": "90%+",
        "off_target_profile": "Moderate — depends on guide design; 1-10% at high-homology off-targets",
        "payload_constraint": "~4.2 kb SpCas9 + guide ≈ AAV limit",
        "key_refs": ["Jinek 2012 Science", "Doudna & Charpentier 2014 Science"],
    },
    {
        "name": "Base Editing — CBE (BE4max)",
        "category": "Precision Editing",
        "mechanism": "Cytosine deamination (C→T / G→A) via nCas9-cytidine deaminase",
        "pam": "NGG (SpCas9-based)",
        "size_aa": "~2000 (nCas9 + APOBEC + UGI)",
        "editing_window": "Positions 4-8 in protospacer (counting from PAM-distal end)",
        "max_efficiency": "60-80%",
        "off_target_profile": "Low genomic off-targets; bystander C editing in window 1-20%",
        "payload_constraint": "~5.2 kb — requires dual-AAV or mRNA delivery",
        "key_refs": ["Komor 2016 Nature", "Koblan 2018 Nat Biotechnol"],
    },
    {
        "name": "Base Editing — ABE (ABE8e)",
        "category": "Precision Editing",
        "mechanism": "Adenosine deamination (A→G / T→C) via nCas9-TadA*",
        "pam": "NGG (SpCas9-based)",
        "size_aa": "~1800",
        "editing_window": "Positions 4-7 in protospacer",
        "max_efficiency": "50-70%",
        "off_target_profile": "Very low genomic off-targets; RNA off-targets reported but transient",
        "payload_constraint": "~5.0 kb",
        "key_refs": ["Gaudelli 2017 Nature", "Richter 2020 Nat Biotechnol"],
    },
    {
        "name": "Prime Editing (PE5max / PEmax)",
        "category": "Precision Editing",
        "mechanism": "Reverse transcription from pegRNA template by nCas9-RT fusion",
        "pam": "NGG",
        "size_aa": "~2200",
        "editing_window": "Flexible: can install any point mutation, small ins/del within ~40 bp",
        "max_efficiency": "30-60% (PE5max in optimised conditions)",
        "off_target_profile": "Very low — no DSB, no deaminase",
        "payload_constraint": "~6.3 kb — challenging for AAV; mRNA/LNP preferred",
        "key_refs": ["Anzalone 2019 Nature", "Chen 2021 Cell", "Doman 2023 Nat Biotechnol"],
    },
    {
        "name": "RNA Editing (ADAR-based)",
        "category": "RNA Editing",
        "mechanism": "A-to-I (read as G) editing on mRNA via engineered ADAR",
        "pam": "N/A (guide RNA directs)",
        "size_aa": "~300-900 (ADAR deaminase domain / full ADAR2)",
        "editing_window": "Target adenosine in dsRNA structure",
        "max_efficiency": "40-80% (on select targets)",
        "off_target_profile": "Transcriptome-wide A-to-I 0.01-1%; transient — no genome alteration",
        "payload_constraint": "Small — ASO or guide RNA only for endogenous ADAR recruitment",
        "key_refs": ["Merkle 2019 Nat Biotechnol", "Qu 2019 Nat Biotechnol"],
    },
    {
        "name": "Cas13 (RNA knockdown)",
        "category": "RNA Targeting",
        "mechanism": "RNA cleavage by Cas13 (HEPN domains)",
        "pam": "PFS: H (non-G) for Cas13a",
        "size_aa": 967,
        "editing_window": "N/A (RNA cleavage)",
        "max_efficiency": "70-95% knockdown",
        "off_target_profile": "Collateral RNA cleavage in vitro; minimal in vivo",
        "payload_constraint": "Small Cas13d (CasRx, 967 aa) fits single AAV",
        "key_refs": ["Abudayyeh 2017 Nature", "Konermann 2018 Cell"],
    },
    {
        "name": "Epigenome Editing (CRISPRi/a)",
        "category": "Epigenetic Modulation",
        "mechanism": "dCas9 fused to KRAB (repressor) or VPR/VP64 (activator)",
        "pam": "NGG",
        "size_aa": "~1600-2100",
        "editing_window": "TSS ± 500 bp for CRISPRi; enhancer regions for CRISPRa",
        "max_efficiency": "80-99% knockdown (CRISPRi); 10-1000× activation (CRISPRa)",
        "off_target_profile": "Very low genomic risk (no DNA modification); some off-target gene regulation",
        "payload_constraint": "~5-6 kb",
        "key_refs": ["Gilbert 2013 Cell", "Kampmann 2018 ACS Chem Biol"],
    },
    {
        "name": "TALEN",
        "category": "Nuclease (DSB)",
        "mechanism": "FokI nuclease directed by TAL effector DNA-binding domains",
        "pam": "5' T preferred",
        "size_aa": "~950 per monomer",
        "editing_window": "N/A (DSB between binding sites, 12-20 bp spacer)",
        "max_efficiency": "20-60%",
        "off_target_profile": "Low — high specificity from extended binding",
        "payload_constraint": "~3 kb per monomer; pair ~6 kb",
        "key_refs": ["Miller 2011 Nat Biotechnol", "Boch 2009 Science"],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Clinical Trials
# ─────────────────────────────────────────────────────────────────────────────
CLINICAL_TRIALS: List[Dict[str, Any]] = [
    {"nct_id": "NCT03745287", "name": "CTX001 / Casgevy", "phase": "Phase III (Approved)", "technology": "CRISPR-Cas9 KO", "target_gene": "BCL11A enhancer", "disease": "Sickle Cell Disease / Beta-Thalassemia", "sponsor": "Vertex / CRISPR Therapeutics", "status": "FDA Approved (Dec 2023)", "year": 2023},
    {"nct_id": "NCT05885464", "name": "Casgevy (β-Thal)", "phase": "Phase III (Approved)", "technology": "CRISPR-Cas9 KO", "target_gene": "BCL11A enhancer", "disease": "Transfusion-dependent Beta-Thalassemia", "sponsor": "Vertex / CRISPR Therapeutics", "status": "FDA Approved (Jan 2024)", "year": 2024},
    {"nct_id": "NCT04774536", "name": "Lyfgenia (Lovo-cel)", "phase": "Phase III (Approved)", "technology": "Lentiviral gene addition", "target_gene": "βA-T87Q-globin", "disease": "Sickle Cell Disease", "sponsor": "bluebird bio", "status": "FDA Approved (Dec 2023)", "year": 2023},
    {"nct_id": "NCT05398029", "name": "VERVE-101", "phase": "Phase Ib", "technology": "Base Editing (ABE)", "target_gene": "PCSK9", "disease": "Heterozygous Familial Hypercholesterolemia", "sponsor": "Verve Therapeutics", "status": "Active — LDL reduction 39-55%", "year": 2023},
    {"nct_id": "NCT04601051", "name": "NTLA-2001", "phase": "Phase I", "technology": "CRISPR-Cas9 KO (in vivo LNP)", "target_gene": "TTR", "disease": "ATTR Amyloidosis", "sponsor": "Intellia Therapeutics", "status": "Active — 87% TTR reduction", "year": 2021},
    {"nct_id": "NCT04560790", "name": "EDIT-101", "phase": "Phase I/II", "technology": "CRISPR-Cas9 (subretinal AAV5)", "target_gene": "CEP290 IVS26", "disease": "Leber Congenital Amaurosis 10", "sponsor": "Editas Medicine", "status": "Active", "year": 2020},
    {"nct_id": "NCT06031727", "name": "BEAM-301", "phase": "Phase I/II", "technology": "Base Editing (ABE)", "target_gene": "R83C in G6PC1", "disease": "Glycogen Storage Disease Ia", "sponsor": "Beam Therapeutics", "status": "Enrolling", "year": 2024},
    {"nct_id": "NCT05456880", "name": "VERVE-102", "phase": "Phase I", "technology": "Base Editing (ABE)", "target_gene": "PCSK9", "disease": "HeFH (GalNAc-LNP)", "sponsor": "Verve Therapeutics", "status": "Active", "year": 2024},
    {"nct_id": "NCT04438083", "name": "CTX110", "phase": "Phase I/II", "technology": "CRISPR-Cas9 KO", "target_gene": "TRAC/B2M", "disease": "B-cell malignancies (allo CAR-T)", "sponsor": "CRISPR Therapeutics", "status": "Active", "year": 2020},
    {"nct_id": "NCT03655678", "name": "CRISPR-HPK", "phase": "Phase I", "technology": "CRISPR-Cas9 KO", "target_gene": "HPK1", "disease": "Refractory cancers (enhanced CAR-T)", "sponsor": "Various", "status": "Active", "year": 2018},
    {"nct_id": "NCT05143307", "name": "WVE-006", "phase": "Phase I/II", "technology": "RNA Editing (endogenous ADAR)", "target_gene": "SERPINA1 (Z allele)", "disease": "Alpha-1 Antitrypsin Deficiency", "sponsor": "Wave Life Sciences", "status": "Active", "year": 2023},
    {"nct_id": "NCT06245005", "name": "Prime Medicine PM359", "phase": "Phase I/II", "technology": "Prime Editing", "target_gene": "CDKL5", "disease": "CDKL5 Deficiency Disorder", "sponsor": "Prime Medicine", "status": "IND Filed 2025", "year": 2025},
    {"nct_id": "NCT06576778", "name": "NTLA-3001", "phase": "Phase I", "technology": "CRISPR-Cas9 KO (in vivo LNP)", "target_gene": "KLKB1 (kallikrein)", "disease": "Hereditary Angioedema", "sponsor": "Intellia Therapeutics", "status": "Active", "year": 2024},
]

# ─────────────────────────────────────────────────────────────────────────────
# 3. Regulatory Approvals
# ─────────────────────────────────────────────────────────────────────────────
REGULATORY_APPROVALS: List[Dict[str, Any]] = [
    {"drug": "Casgevy (exagamglogene autotemcel)", "technology": "CRISPR-Cas9 KO", "target": "BCL11A enhancer in autologous CD34+ HSCs", "indication": "Sickle Cell Disease (≥12 yr, recurrent VOC)", "agency": "FDA", "date": "2023-12-08", "note": "First CRISPR therapy approved by FDA"},
    {"drug": "Casgevy", "technology": "CRISPR-Cas9 KO", "target": "BCL11A enhancer", "indication": "Transfusion-Dependent Beta-Thalassemia (≥12 yr)", "agency": "FDA", "date": "2024-01-16", "note": "Second indication"},
    {"drug": "Casgevy", "technology": "CRISPR-Cas9 KO", "target": "BCL11A enhancer", "indication": "SCD + TDT", "agency": "EMA", "date": "2024-02-01", "note": "EU conditional approval"},
    {"drug": "Casgevy", "technology": "CRISPR-Cas9 KO", "target": "BCL11A enhancer", "indication": "SCD + TDT", "agency": "MHRA (UK)", "date": "2023-11-16", "note": "World-first CRISPR approval"},
]

# ─────────────────────────────────────────────────────────────────────────────
# 4. Technology–Disease Matrix
# ─────────────────────────────────────────────────────────────────────────────
TECH_DISEASE_MATRIX: List[Dict[str, Any]] = [
    {"technology": "CRISPR-Cas9 KO", "disease": "Sickle Cell Disease", "target_gene": "BCL11A", "mutation_strategy": "Enhancer disruption → reactivate HbF", "evidence_level": "Level 1 (FDA Approved)", "best_efficiency": "95% HbF induction", "key_paper_doi": "10.1056/NEJMoa2031054"},
    {"technology": "CRISPR-Cas9 KO", "disease": "Beta-Thalassemia", "target_gene": "BCL11A", "mutation_strategy": "Enhancer disruption → reactivate HbF", "evidence_level": "Level 1 (FDA Approved)", "best_efficiency": "~95%", "key_paper_doi": "10.1056/NEJMoa2031054"},
    {"technology": "Base Editing ABE", "disease": "Sickle Cell Disease", "target_gene": "HBB", "mutation_strategy": "Direct correction of E6V (GAG→GTG, A→G edit)", "evidence_level": "Level 2 (In vivo mouse)", "best_efficiency": "58-68% in HSC", "key_paper_doi": "10.1038/s41586-021-03609-w"},
    {"technology": "Base Editing ABE", "disease": "Hypercholesterolemia (HeFH)", "target_gene": "PCSK9", "mutation_strategy": "Splice-site disruption of PCSK9", "evidence_level": "Level 1 (Phase Ib)", "best_efficiency": "55% LDL reduction", "key_paper_doi": "10.1056/NEJMoa2303223"},
    {"technology": "CRISPR-Cas9 KO", "disease": "ATTR Amyloidosis", "target_gene": "TTR", "mutation_strategy": "In vivo LNP KO of TTR in liver", "evidence_level": "Level 1 (Phase I)", "best_efficiency": "87% TTR reduction", "key_paper_doi": "10.1056/NEJMoa2107454"},
    {"technology": "Prime Editing", "disease": "Sickle Cell Disease", "target_gene": "HBB", "mutation_strategy": "Direct E6V correction (versatile edit)", "evidence_level": "Level 2 (In vivo mouse)", "best_efficiency": "30-40% in HSC", "key_paper_doi": "10.1038/s41586-023-06004-3"},
    {"technology": "CRISPR-Cas9 KO", "disease": "Leber Congenital Amaurosis 10", "target_gene": "CEP290 IVS26", "mutation_strategy": "Excise intronic mutation (subretinal AAV)", "evidence_level": "Level 1 (Phase I/II)", "best_efficiency": "Improved visual acuity", "key_paper_doi": "10.1056/NEJMoa2309915"},
    {"technology": "RNA Editing ADAR", "disease": "Alpha-1 Antitrypsin Deficiency", "target_gene": "SERPINA1", "mutation_strategy": "Correct Z-allele E342K at mRNA level", "evidence_level": "Level 1 (Phase I/II)", "best_efficiency": "~40% editing at target site", "key_paper_doi": "N/A (WVE-006)"},
    {"technology": "CRISPR-Cas9 KO", "disease": "Duchenne Muscular Dystrophy", "target_gene": "DMD exon 51/53", "mutation_strategy": "Exon skipping to restore reading frame", "evidence_level": "Level 2 (Animal)", "best_efficiency": "Dystrophin restoration 5-50%", "key_paper_doi": "10.1126/science.aad5143"},
    {"technology": "Base Editing CBE", "disease": "Beta-Thalassemia", "target_gene": "HBG1/HBG2 promoter", "mutation_strategy": "Recreate HPFH mutations → reactivate HbF", "evidence_level": "Level 2 (In vivo mouse)", "best_efficiency": "60-80% HbF induction", "key_paper_doi": "10.1182/blood.2020009674"},
    {"technology": "CRISPR-Cas9 KO", "disease": "HIV", "target_gene": "CCR5", "mutation_strategy": "CCR5Δ32-like disruption in HSC/T-cells", "evidence_level": "Level 1 (Phase I)", "best_efficiency": "5-8% CCR5-null in infused cells", "key_paper_doi": "10.1056/NEJMoa1817426"},
    {"technology": "CRISPR-Cas9 KO", "disease": "Hereditary Angioedema", "target_gene": "KLKB1", "mutation_strategy": "In vivo LNP KO of kallikrein", "evidence_level": "Level 1 (Phase I)", "best_efficiency": ">90% kallikrein reduction", "key_paper_doi": "N/A (NTLA-3001)"},
]

# ─────────────────────────────────────────────────────────────────────────────
# 5. Safety Profiles
# ─────────────────────────────────────────────────────────────────────────────
SAFETY_PROFILES: List[Dict[str, Any]] = [
    {"technology": "CRISPR-Cas9 KO", "context": "Ex vivo HSC", "off_target_genomic": "0.1-5% at top predicted sites", "off_target_method": "GUIDE-seq, CIRCLE-seq, Digenome-seq", "immunogenicity": "Low (pre-existing anti-Cas9 Ab in 58-79% of humans; mitigated by transient RNP)", "chromosomal_abnormality": "Rare chromothripsis reported (Leibowitz 2021 Nat Genet)", "bystander_editing": "N/A", "clinical_safety": "Casgevy: myeloablative conditioning risk; 1 death (VOC, not attributed to editing)"},
    {"technology": "Base Editing ABE", "context": "In vivo liver (LNP)", "off_target_genomic": "<0.01% (ABE8e optimised)", "off_target_method": "ONE-seq, Digenome-seq", "immunogenicity": "Low (transient mRNA)", "chromosomal_abnormality": "Not observed", "bystander_editing": "A-to-G at adjacent adenines in editing window (1-20%)", "clinical_safety": "VERVE-101: no SAE at 12mo; transient transaminase elevation"},
    {"technology": "Base Editing CBE", "context": "Ex vivo / in vivo", "off_target_genomic": "0.01-0.1%", "off_target_method": "GUIDE-seq adapted", "immunogenicity": "Low", "chromosomal_abnormality": "Not observed", "bystander_editing": "C-to-T at adjacent cytosines in editing window (5-30%)", "clinical_safety": "Limited clinical data (BEAM-301 IND)"},
    {"technology": "Prime Editing", "context": "Ex vivo / in vivo", "off_target_genomic": "<0.01% (no DSB, no deaminase)", "off_target_method": "PEM-seq, targeted amplicon sequencing", "immunogenicity": "Low (mRNA delivery)", "chromosomal_abnormality": "Not observed", "bystander_editing": "Scaffold insertion artifacts (1-5%, mitigated by PE5max)", "clinical_safety": "Pre-clinical; IND-enabling studies (Prime Medicine 2025)"},
    {"technology": "RNA Editing ADAR", "context": "In vivo (ASO delivery)", "off_target_genomic": "None (no DNA modification)", "off_target_method": "RNA-seq transcriptome-wide", "immunogenicity": "Minimal", "chromosomal_abnormality": "None", "bystander_editing": "Transcriptome-wide A-to-I: 0.01-1% (dose-dependent)", "clinical_safety": "Wave WVE-006: well-tolerated in Phase I/II"},
]

# ─────────────────────────────────────────────────────────────────────────────
# 6. Guide RNA Design Resources
# ─────────────────────────────────────────────────────────────────────────────
GRNA_DATABASES: List[Dict[str, str]] = [
    {"name": "CRISPRko Library (Brunello)", "url": "https://www.addgene.org/pooled-library/broadgpp-human-knockout-brunello/", "scope": "Human genome-wide KO (77,441 guides)"},
    {"name": "CRISPick (Broad)", "url": "https://portals.broadinstitute.org/gppx/crispick/a", "scope": "Custom guide design for Cas9/Cas12a"},
    {"name": "BE-Hive", "url": "https://www.crisprbehive.design", "scope": "Base editing outcome prediction (CBE/ABE)"},
    {"name": "PrimeDesign", "url": "https://primedesign.pinellolab.partners.org/", "scope": "pegRNA design for Prime Editing"},
    {"name": "Cas-OFFinder", "url": "http://www.rgenome.net/cas-offinder/", "scope": "Genome-wide off-target search"},
    {"name": "CHOPCHOP", "url": "https://chopchop.cbu.uib.no/", "scope": "Multi-purpose guide design (Cas9/Cas12/Cas13)"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Query Interface
# ─────────────────────────────────────────────────────────────────────────────
class GEAlmanac:
    """Query interface for the Gene Editing Almanac structured knowledge base."""

    def __init__(self, extra_data_path: Optional[str] = None):
        self.tech_registry = TECHNOLOGY_REGISTRY
        self.clinical_trials = CLINICAL_TRIALS
        self.approvals = REGULATORY_APPROVALS
        self.tech_disease = TECH_DISEASE_MATRIX
        self.safety = SAFETY_PROFILES
        self.grna_dbs = GRNA_DATABASES
        # Load optional user-extended data
        if extra_data_path and os.path.exists(extra_data_path):
            try:
                with open(extra_data_path, "r", encoding="utf-8") as f:
                    extra = json.load(f)
                self.clinical_trials.extend(extra.get("clinical_trials", []))
                self.tech_disease.extend(extra.get("tech_disease", []))
                print(f"[GEAlmanac] Loaded extra data from {extra_data_path}")
            except Exception as e:
                print(f"[GEAlmanac] Failed to load extra data: {e}")

    # ── Technology queries ────────────────────────────────────────────────

    def get_technology(self, name: str) -> Optional[Dict]:
        """Fuzzy-match a technology by name."""
        nl = name.lower()
        for t in self.tech_registry:
            if nl in t["name"].lower() or any(
                nl in kw for kw in [t.get("category", "").lower(), t.get("mechanism", "").lower()]
            ):
                return t
        return None

    def list_technologies(self) -> List[str]:
        return [t["name"] for t in self.tech_registry]

    # ── Clinical trial queries ────────────────────────────────────────────

    def query_trials(
        self,
        disease: Optional[str] = None,
        technology: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> List[Dict]:
        results = self.clinical_trials
        if disease:
            dl = disease.lower()
            results = [t for t in results if dl in t.get("disease", "").lower()]
        if technology:
            tl = technology.lower()
            results = [t for t in results if tl in t.get("technology", "").lower()]
        if phase:
            pl = phase.lower()
            results = [t for t in results if pl in t.get("phase", "").lower()]
        return results

    # ── Regulatory queries ────────────────────────────────────────────────

    def query_approvals(self, technology: Optional[str] = None) -> List[Dict]:
        if not technology:
            return self.approvals
        tl = technology.lower()
        return [a for a in self.approvals if tl in a.get("technology", "").lower()]

    # ── Technology–Disease matrix ─────────────────────────────────────────

    def recommend_for_disease(self, disease: str) -> List[Dict]:
        """Return all technologies studied for a given disease, sorted by evidence level."""
        dl = disease.lower()
        matches = [m for m in self.tech_disease if dl in m.get("disease", "").lower()]
        level_order = {"Level 1": 0, "Level 2": 1, "Level 3": 2, "Level 4": 3}
        matches.sort(key=lambda x: level_order.get(x.get("evidence_level", "Level 4")[:7], 9))
        return matches

    def recommend_for_mutation(self, mutation_type: str) -> List[Dict]:
        """Recommend technologies based on mutation type."""
        ml = mutation_type.lower()
        recommendations = []
        if "snv" in ml or "point" in ml:
            recommendations = [
                {"technology": "Base Editing (ABE/CBE)", "reason": "Direct single-base correction without DSB", "evidence": "Level 1-2"},
                {"technology": "Prime Editing", "reason": "Can correct any point mutation; no bystander editing", "evidence": "Level 2"},
                {"technology": "CRISPR-Cas9 + HDR", "reason": "Template-directed correction; lower efficiency", "evidence": "Level 2-3"},
            ]
        elif "insertion" in ml:
            recommendations = [
                {"technology": "Prime Editing", "reason": "Can insert up to ~40 bp via pegRNA template", "evidence": "Level 2"},
                {"technology": "CRISPR-Cas9 + HDR", "reason": "Larger insertions possible with donor template", "evidence": "Level 2-3"},
            ]
        elif "deletion" in ml:
            recommendations = [
                {"technology": "CRISPR-Cas9 KO (dual-guide)", "reason": "Pair of guides flanking region to delete", "evidence": "Level 1-2"},
                {"technology": "Prime Editing", "reason": "Precise deletions up to ~80 bp", "evidence": "Level 2"},
            ]
        elif "exon" in ml or "frameshift" in ml:
            recommendations = [
                {"technology": "CRISPR-Cas9 (exon skipping)", "reason": "Disrupt splice sites to skip mutant exon", "evidence": "Level 2 (DMD)"},
                {"technology": "Base Editing", "reason": "Modify splice donor/acceptor sites", "evidence": "Level 2"},
            ]
        return recommendations

    # ── Safety queries ────────────────────────────────────────────────────

    def get_safety_profile(self, technology: str, context: Optional[str] = None) -> List[Dict]:
        tl = technology.lower()
        matches = [s for s in self.safety if tl in s["technology"].lower()]
        if context:
            cl = context.lower()
            filtered = [s for s in matches if cl in s.get("context", "").lower()]
            if filtered:
                return filtered
        return matches

    # ── Formatted report for LLM context ──────────────────────────────────

    def generate_almanac_context(self, query: str) -> str:
        """Generate a structured almanac report for injection into LLM context."""
        q = query.lower()
        sections = []

        # Disease-specific recommendations
        disease_keywords = {
            "sickle cell": "Sickle Cell Disease", "scd": "Sickle Cell Disease",
            "thalassemia": "Beta-Thalassemia", "β-thal": "Beta-Thalassemia",
            "amyloidosis": "ATTR Amyloidosis", "attr": "ATTR Amyloidosis",
            "hypercholesterolemia": "Hypercholesterolemia", "hef": "Hypercholesterolemia",
            "pcsk9": "Hypercholesterolemia",
            "lca": "Leber Congenital Amaurosis", "leber": "Leber Congenital Amaurosis",
            "dmd": "Duchenne Muscular Dystrophy", "duchenne": "Duchenne Muscular Dystrophy",
            "hiv": "HIV", "ccr5": "HIV",
        }
        for kw, disease in disease_keywords.items():
            if kw in q:
                recs = self.recommend_for_disease(disease)
                if recs:
                    sections.append(f"### GEAlmanac: {disease}")
                    for r in recs:
                        sections.append(
                            f"- **{r['technology']}** → {r['target_gene']}: "
                            f"{r['mutation_strategy']} | Evidence: {r['evidence_level']} | "
                            f"Best efficiency: {r['best_efficiency']}"
                        )
                # Clinical trials
                trials = self.query_trials(disease=disease)
                if trials:
                    sections.append(f"\n**Active Clinical Trials for {disease}:**")
                    for t in trials[:5]:
                        sections.append(
                            f"- [{t['nct_id']}] {t['name']} ({t['phase']}) — "
                            f"{t['technology']} targeting {t['target_gene']} | {t['status']}"
                        )
                break  # Only match first disease

        # Technology-specific info
        tech_keywords = {
            "base editing": "Base Editing", "abe": "ABE", "cbe": "CBE",
            "prime editing": "Prime Editing", "crispr": "CRISPR-Cas9",
            "cas9": "CRISPR-Cas9", "cas13": "Cas13",
            "rna editing": "RNA Editing", "adar": "RNA Editing",
        }
        for kw, tech in tech_keywords.items():
            if kw in q:
                safety = self.get_safety_profile(tech)
                if safety:
                    sections.append(f"\n### GEAlmanac Safety: {tech}")
                    for s in safety[:2]:
                        sections.append(
                            f"- Context: {s['context']}\n"
                            f"  Off-target (genomic): {s['off_target_genomic']}\n"
                            f"  Method: {s['off_target_method']}\n"
                            f"  Bystander: {s['bystander_editing']}\n"
                            f"  Clinical: {s['clinical_safety']}"
                        )
                break

        # Regulatory approvals
        if any(kw in q for kw in ["approved", "fda", "ema", "regulatory", "批准"]):
            sections.append("\n### GEAlmanac: Regulatory Approvals")
            for a in self.approvals:
                sections.append(
                    f"- **{a['drug']}** | {a['indication']} | {a['agency']} {a['date']} | {a['note']}"
                )

        return "\n".join(sections) if sections else ""
