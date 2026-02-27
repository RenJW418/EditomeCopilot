"""
Failure Case Database
=====================
Curated database of gene editing failures, terminated trials, safety events,
retracted papers, delivery failures, and technology bottlenecks.

Purpose: When the system generates an optimistic recommendation, it cross-checks
against known failure modes and provides "similar failure" warnings.

Data sources: ClinicalTrials.gov terminations, FDA black-box warnings,
retracted publications (Retraction Watch), published adverse events.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Terminated or Paused Clinical Trials
# ─────────────────────────────────────────────────────────────────────────────
TERMINATED_TRIALS: List[Dict[str, Any]] = [
    {
        "trial_id": "NCT03041324",
        "title": "CasgevyTM EX-istence — CLIMB-121 (Severe Adverse Event note)",
        "sponsor": "Vertex/CRISPR Therapeutics",
        "technology": "CRISPR-Cas9",
        "target": "BCL11A enhancer in HSCs",
        "disease": "Sickle Cell Disease / Beta-Thalassemia",
        "status": "Completed (with SAE notes)",
        "failure_category": "safety_event",
        "description": "One patient death attributed to myeloablative conditioning (busulfan), not directly to gene editing. Highlights: myeloablative conditioning remains the major safety bottleneck for ex vivo HSC editing.",
        "lesson": "Ex vivo HSC therapies inherently carry myeloablative conditioning risk. Non-myeloablative conditioning approaches (e.g., antibody-drug conjugates targeting CD117) are needed.",
        "date": "2024-01",
        "refs": ["10.1056/NEJMoa2031054"],
    },
    {
        "trial_id": "NCT04601051",
        "title": "EDIT-301 for SCD",
        "sponsor": "Editas Medicine",
        "technology": "CRISPR-Cas12a (AsCas12a)",
        "target": "HBG1/HBG2 promoter in HSCs",
        "disease": "Sickle Cell Disease",
        "status": "Terminated (2024-06)",
        "failure_category": "commercial_strategic",
        "description": "Terminated due to competitive landscape (Casgevy approval) despite positive Phase I/II data. Not a safety or efficacy failure per se.",
        "lesson": "First-mover advantage in rare disease gene therapy is critical. Late entrants face reimbursement and market access barriers even with similar efficacy.",
        "date": "2024-06",
        "refs": [],
    },
    {
        "trial_id": "NCT03872479",
        "title": "BRILLIANCE — EDIT-101 for LCA10",
        "sponsor": "Editas Medicine",
        "technology": "CRISPR-Cas9 (AAV5-delivered in vivo)",
        "target": "CEP290 IVS26 intronic mutation",
        "disease": "Leber Congenital Amaurosis 10",
        "status": "Active (dose-escalation concerns)",
        "failure_category": "efficacy_challenge",
        "description": "Initial low/mid doses showed limited efficacy. Higher doses showed some visual improvement but AAV immune responses detected. Subretinal AAV5 delivery limits transduced cell population.",
        "lesson": "In vivo AAV delivery for retinal editing has dose-response challenges. Immune responses to AAV capsid limit re-dosing. Pre-existing anti-AAV antibodies exclude some patients.",
        "date": "2024-03",
        "refs": ["10.1056/NEJMoa2309915"],
    },
    {
        "trial_id": "NCT05398029",
        "title": "EBT-101 for HIV",
        "sponsor": "Excision BioTherapeutics",
        "technology": "CRISPR-Cas9 (multiplex, AAV9)",
        "target": "HIV proviral DNA (gag, pol, LTR)",
        "disease": "HIV-1",
        "status": "Phase I (cautious progress)",
        "failure_category": "delivery_challenge",
        "description": "Multiplex CRISPR targeting proviral DNA requires systemic delivery to all reservoir cells (latent CD4+ T cells, macrophages). AAV9 tropism primarily neurotropic, suboptimal for lymphoid tissue. Three-guide approach increases off-target risk.",
        "lesson": "HIV cure via CRISPR faces fundamental delivery challenge: reaching all latent reservoir cells throughout the body. Single-dose cure unlikely with current delivery technology.",
        "date": "2023-09",
        "refs": [],
    },
    {
        "trial_id": "Placeholder-PCSK9-001",
        "title": "VERVE-101 Phase Ib Dose-Response Concerns",
        "sponsor": "Verve Therapeutics",
        "technology": "ABE (LNP-delivered in vivo)",
        "target": "PCSK9 splice site",
        "disease": "HeFH",
        "status": "Phase Ib (dose-dependent adverse events detected)",
        "failure_category": "safety_event",
        "description": "At highest dose (0.6 mg/kg), one patient experienced hepatic SAE (ALT/AST elevation). Subsequently, one patient death (cardiovascular event, deemed unrelated but trial paused for review). Stock dropped ~40%.",
        "lesson": "LNP dose-response has a narrow therapeutic window for liver editing. Hepatotoxicity at higher doses constrains maximum achievable editing efficiency.",
        "date": "2024-02",
        "refs": ["10.1056/NEJMoa2303223"],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Known Safety Events / Adverse Outcomes
# ─────────────────────────────────────────────────────────────────────────────
SAFETY_EVENTS: List[Dict[str, Any]] = [
    {
        "event_id": "SE-001",
        "category": "chromosomal_abnormality",
        "technology": "CRISPR-Cas9 (DSB-based)",
        "description": "Chromothripsis detected in human embryo editing experiments. DSBs at on-target site led to large deletions, translocations, and chromothripsis on chromosome 6.",
        "affected_targets": ["embryo editing", "any DSB-based approach"],
        "severity": "HIGH",
        "mitigation": "Use DSB-free editors (base editing, prime editing) when possible. Comprehensive cytogenetic screening required for DSB-based therapies.",
        "refs": ["10.1016/j.cell.2020.11.040", "Zuccaro et al. 2020 Cell"],
        "date": "2020-11",
    },
    {
        "event_id": "SE-002",
        "category": "off_target_editing",
        "technology": "CBE (APOBEC-based)",
        "description": "Genome-wide off-target C→T editing detected in mouse embryos and rice. APOBEC deaminase domain acts on single-stranded DNA genome-wide, not just at guide-directed sites. Transcriptome-wide C→U editing on RNA also detected.",
        "affected_targets": ["any CBE-edited cells"],
        "severity": "MEDIUM-HIGH",
        "mitigation": "Use engineered APOBEC variants (YE1, eA3A) with narrowed window. Perform whole-genome sequencing. RNA off-targets mitigable with SECURE-CBE.",
        "refs": ["10.1126/science.aav5892", "10.1038/s41586-019-1314-0"],
        "date": "2019-03",
    },
    {
        "event_id": "SE-003",
        "category": "immune_response_aav",
        "technology": "AAV-delivered CRISPR (any)",
        "description": "Pre-existing anti-AAV antibodies in 30-60% of human population neutralize AAV vectors. Post-administration, anti-capsid T cell responses can destroy transduced hepatocytes. Re-dosing impossible with same serotype.",
        "affected_targets": ["any AAV-delivered gene editing"],
        "severity": "MEDIUM",
        "mitigation": "Screen for anti-AAV antibodies pre-enrollment. Use engineered capsids (e.g., Anc80L65). Switch serotypes for re-dosing. Consider non-viral LNP delivery.",
        "refs": ["10.1038/s41591-020-0847-9"],
        "date": "2020-05",
    },
    {
        "event_id": "SE-004",
        "category": "p53_selection",
        "technology": "CRISPR-Cas9 (DSB-based)",
        "description": "DSBs activate p53 pathway → cells with pre-existing TP53 mutations gain survival advantage → CRISPR-edited cell populations enriched for p53-deficient (pre-malignant) cells.",
        "affected_targets": ["any DSB-based editing in proliferating cells"],
        "severity": "MEDIUM-HIGH",
        "mitigation": "Use DSB-free editors. If using Cas9 nuclease, perform TP53 status monitoring of edited cell products. Transient p53 inhibition during editing (controversial).",
        "refs": ["10.1038/s41591-018-0049-z", "Haapaniemi et al. 2018 Nat Med"],
        "date": "2018-06",
    },
    {
        "event_id": "SE-005",
        "category": "immune_response_cas",
        "technology": "SpCas9 (any delivery)",
        "description": "Pre-existing adaptive immunity to SpCas9 detected in 58-78% of humans (from natural Staphylococcus aureus / Streptococcus pyogenes exposure). Anti-Cas9 T cells and antibodies may clear edited cells in vivo.",
        "affected_targets": ["Any SpCas9-based in vivo therapy"],
        "severity": "MEDIUM",
        "mitigation": "Use transient mRNA/RNP delivery (Cas9 protein degrades in ~48h). Use orthogonal Cas proteins (CjCas9, Cas12a) with lower human pre-existing immunity. Immunosuppression regimen.",
        "refs": ["10.1038/s41591-019-0459-6", "Charlesworth et al. 2019 Nat Med"],
        "date": "2019-04",
    },
    {
        "event_id": "SE-006",
        "category": "lnp_toxicity",
        "technology": "LNP-delivered mRNA (any)",
        "description": "LNP components (ionizable lipids) cause dose-dependent liver inflammation (transaminase elevation). At higher doses, complement activation (CARPA) and cytokine release possible. Observed in VERVE-101 and multiple preclinical studies.",
        "affected_targets": ["Any LNP-delivered gene editing"],
        "severity": "MEDIUM",
        "mitigation": "Optimize LNP composition (biodegradable ionizable lipids). Pre-medicate with dexamethasone. Minimize dose via higher-activity editors. Consider alternative delivery (VLP, exosomes).",
        "refs": ["10.1038/s41578-021-00358-0"],
        "date": "2023-01",
    },
    {
        "event_id": "SE-007",
        "category": "prime_editing_insert",
        "technology": "Prime Editing",
        "description": "Unintended insertions of PE scaffold sequence at the target site observed in 1-10% of edits. RT can read through the pegRNA scaffold, inserting extra sequence at the edit site.",
        "affected_targets": ["Any prime editing application"],
        "severity": "LOW-MEDIUM",
        "mitigation": "Use engineered pegRNAs with structured 3' end (e.g., tevopreQ1 motif — epegRNA) to block RT read-through. Screen clones for scaffold insertion.",
        "refs": ["10.1038/s41587-021-01039-7", "Nelson et al. 2022 Nat Biotechnol"],
        "date": "2022-01",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Technology Bottlenecks
# ─────────────────────────────────────────────────────────────────────────────
TECHNOLOGY_BOTTLENECKS: List[Dict[str, Any]] = [
    {
        "bottleneck_id": "BN-001",
        "technology": "Base Editing (all)",
        "bottleneck": "Bystander editing in activity window",
        "description": "All base editors edit C/A within a 4-8 nt window. If additional C/A bases exist near the target, they'll be edited (bystander edits) — potentially creating pathogenic mutations.",
        "impact": "Limits applicability to loci where target base is the ONLY editable base in window. ~30% of pathogenic SNVs have bystander concerns.",
        "solutions_in_progress": "Narrowed-window variants (BE4-YE1, ABE8e-V106W); PE as alternative",
    },
    {
        "bottleneck_id": "BN-002",
        "technology": "Prime Editing",
        "bottleneck": "Lower efficiency vs. base editors",
        "description": "PE typically achieves 20-50% editing in dividing cells, 5-20% in post-mitotic cells. Compared to ABE (50-70%) and CBE (60-80%). PE also requires larger payload (PE2: 6.3kb + pegRNA + nicking guide).",
        "impact": "For applications requiring high editing efficiency (e.g., in vivo liver), PE may be insufficient. Payload size limits AAV delivery.",
        "solutions_in_progress": "PE5max, PE7; split-intein dual-AAV; mRNA/LNP delivery; PEmax architecture",
    },
    {
        "bottleneck_id": "BN-003",
        "technology": "In vivo delivery (all)",
        "bottleneck": "Tissue-specific targeting beyond liver",
        "description": "LNP overwhelmingly targets liver (>90% of dose). Non-liver tissues (brain, muscle, lung, heart) lack efficient non-viral delivery. AAV provides broader tropism but has size limits and immunogenicity.",
        "impact": "This is THE major bottleneck for most gene editing therapies. ~70% of genetic diseases affect non-liver tissues.",
        "solutions_in_progress": "Engineered LNP (SORT, SEND); tissue-specific AAV capsids (MyoAAV, AAV.PHP.eB); VLPs; exosomes; cell-specific antibody-LNP conjugates",
    },
    {
        "bottleneck_id": "BN-004",
        "technology": "Ex vivo HSC editing",
        "bottleneck": "Myeloablative conditioning requirement",
        "description": "Current ex vivo HSC therapies (Casgevy, Lyfgenia) require full myeloablative conditioning (busulfan). This is toxic, requires prolonged hospitalization, and carries non-trivial mortality risk (1-5%).",
        "impact": "Limits accessibility to major medical centers. Patient eligibility restricted. Cost >$2M per patient partly due to conditioning/hospitalization.",
        "solutions_in_progress": "Antibody-based conditioning (anti-CD117/cKIT — JSP191); reduced-intensity conditioning; in vivo HSC editing (mobilized + LNP)",
    },
    {
        "bottleneck_id": "BN-005",
        "technology": "Multiplex editing",
        "bottleneck": "Translocation risk with multiple DSBs",
        "description": "Simultaneous DSBs at multiple loci create risk of chromosome translocations between cut sites. Risk scales quadratically with number of cuts.",
        "impact": "Limits safe number of simultaneous edits to 2-3 with DSB-based approaches. Major concern for allogeneic CAR-T manufacturing (often needs 3-4 KOs).",
        "solutions_in_progress": "Base editing for multiplex KO (no DSB); sequential editing; Cas12a multiplex arrays; cytidine base editor for multiplex splice KO",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Retracted / Controversial Publications
# ─────────────────────────────────────────────────────────────────────────────
RETRACTED_PAPERS: List[Dict[str, Any]] = [
    {
        "paper_id": "RP-001",
        "title": "He Jiankui's human embryo editing (CCR5 Δ32)",
        "year": 2018,
        "status": "Retracted / Condemned",
        "journal": "N/A (not published in peer-reviewed journal)",
        "issue": "Edited human embryos implanted without proper ethical oversight, inadequate informed consent, mosaic editing, off-target effects uncharacterized.",
        "impact": "Led to international moratorium on heritable human genome editing. He Jiankui imprisoned (2019-2022).",
        "lesson": "Germline editing technology is NOT ready for clinical application. Mosaicism and off-target characterization remain insufficient.",
    },
    {
        "paper_id": "RP-002",
        "title": "Anti-CRISPR concerns — Cas9 causes large deletions (Kosicki et al. 2018 Nat Biotechnol)",
        "year": 2018,
        "status": "Valid — confirmed by multiple groups",
        "journal": "Nature Biotechnology",
        "issue": "Cas9 DSBs cause large deletions (up to several kb) at on-target site, not just small indels as commonly assumed. These are often missed by standard PCR-based genotyping.",
        "impact": "Changed safety assessment requirements for all DSB-based therapies. Long-range PCR and cytogenetics now required.",
        "lesson": "Standard short-range PCR genotyping is insufficient to characterize Cas9 editing outcomes. Always use long-range sequencing.",
        "ref": "10.1038/nbt.4192",
    },
    {
        "paper_id": "RP-003",
        "title": "CRISPR Babies — Denis Rebrikov proposed germline editing for deafness (GJB2)",
        "year": 2019,
        "status": "Proposed but not executed",
        "journal": "Nature News",
        "issue": "Russian scientist Denis Rebrikov proposed editing GJB2 mutations in embryos to prevent deafness. Universally criticized as ethically unjustified — deafness is not life-threatening and has other interventions.",
        "impact": "Further reinforced need for international governance framework.",
        "lesson": "Germline editing should only be considered for severe diseases with no alternative treatment — NOT for non-life-threatening conditions.",
    },
]


class FailureCaseDB:
    """Query interface for gene editing failure cases."""

    def __init__(self):
        self.trials = TERMINATED_TRIALS
        self.safety = SAFETY_EVENTS
        self.bottlenecks = TECHNOLOGY_BOTTLENECKS
        self.retracted = RETRACTED_PAPERS

    def search(self, query: str, technology: str = None) -> Dict[str, List]:
        """
        Search failure database for relevant cases.

        Args:
            query: Natural language query or keywords
            technology: Optional filter (e.g., "CRISPR-Cas9", "ABE", "LNP")

        Returns:
            Dict with keys: trials, safety_events, bottlenecks, retracted
        """
        q = query.lower()
        results = {
            "trials": [],
            "safety_events": [],
            "bottlenecks": [],
            "retracted": [],
        }

        # Search terminated trials
        for trial in self.trials:
            if self._matches(trial, q, technology):
                results["trials"].append(trial)

        # Search safety events
        for event in self.safety:
            if self._matches(event, q, technology):
                results["safety_events"].append(event)

        # Search bottlenecks
        for bn in self.bottlenecks:
            if self._matches(bn, q, technology):
                results["bottlenecks"].append(bn)

        # Search retracted papers
        for paper in self.retracted:
            if self._matches(paper, q, technology):
                results["retracted"].append(paper)

        return results

    def _matches(self, entry: Dict, q: str, technology: str = None) -> bool:
        """Check if an entry matches query and optional technology filter."""
        entry_text = " ".join(str(v) for v in entry.values()).lower()

        # Technology filter
        if technology:
            tech_lower = technology.lower()
            entry_tech = entry.get("technology", "").lower()
            if tech_lower not in entry_tech and tech_lower not in entry_text:
                return False

        # Query matching — any keyword from query found in entry
        keywords = [w for w in q.split() if len(w) > 2]
        if not keywords:
            return True  # No filter → return all

        return any(kw in entry_text for kw in keywords)

    def get_warnings_for_strategy(self, technology: str, target_tissue: str = "",
                                   disease: str = "") -> List[str]:
        """
        Given a proposed editing strategy, return relevant failure warnings.

        Returns list of warning strings suitable for LLM injection.
        """
        warnings = []
        context = f"{technology} {target_tissue} {disease}".lower()

        # Safety events
        for event in self.safety:
            tech = event.get("technology", "").lower()
            if any(t in context for t in tech.split("(")):
                severity = event.get("severity", "?")
                warnings.append(
                    f"⚠️ [{severity}] {event['category']}: {event['description'][:200]}... "
                    f"Mitigation: {event.get('mitigation', 'N/A')[:150]}"
                )

        # Check for DSB-specific warnings
        if any(k in context for k in ["cas9", "cas12", "nuclease", "dsb", "knockout", "ko"]):
            dsb_warnings = [e for e in self.safety if e["category"] in
                           ("chromosomal_abnormality", "p53_selection")]
            for w in dsb_warnings:
                warning_text = f"⚠️ [{w['severity']}] {w['category']}: {w['description'][:200]}"
                if warning_text not in warnings:
                    warnings.append(warning_text)

        # Delivery-specific warnings
        if "aav" in context:
            aav_warnings = [e for e in self.safety if "aav" in e.get("category", "").lower()
                           or "aav" in e.get("technology", "").lower()]
            for w in aav_warnings:
                warning_text = f"⚠️ [{w['severity']}] {w['category']}: {w['description'][:200]}"
                if warning_text not in warnings:
                    warnings.append(warning_text)

        if "lnp" in context:
            lnp_warnings = [e for e in self.safety if "lnp" in e.get("technology", "").lower()]
            for w in lnp_warnings:
                warning_text = f"⚠️ [{w['severity']}] {w['category']}: {w['description'][:200]}"
                if warning_text not in warnings:
                    warnings.append(warning_text)

        # Bottlenecks
        for bn in self.bottlenecks:
            bn_tech = bn.get("technology", "").lower()
            if any(t in context for t in bn_tech.split("(")[0].lower().split()):
                warnings.append(
                    f"🔧 BOTTLENECK — {bn['bottleneck']}: {bn['description'][:200]}... "
                    f"Solutions: {bn.get('solutions_in_progress', 'N/A')[:150]}"
                )

        return warnings[:5]  # Cap at 5 most relevant

    def format_context(self, query: str, technology: str = None) -> str:
        """Format failure cases as structured text for LLM context injection."""
        results = self.search(query, technology)

        has_results = any(results[k] for k in results)
        if not has_results:
            return ""

        lines = ["### ⚠️ Failure Case & Risk Intelligence"]

        if results["trials"]:
            lines.append("\n**Relevant Terminated/Paused Trials:**")
            for t in results["trials"][:3]:
                lines.append(f"- **{t['title']}** ({t['status']})")
                lines.append(f"  {t['description'][:200]}")
                lines.append(f"  📝 Lesson: {t['lesson'][:150]}")

        if results["safety_events"]:
            lines.append("\n**Known Safety Events:**")
            for e in results["safety_events"][:3]:
                lines.append(f"- **[{e['severity']}] {e['category']}** — {e['description'][:200]}")
                lines.append(f"  Mitigation: {e['mitigation'][:150]}")

        if results["bottlenecks"]:
            lines.append("\n**Technology Bottlenecks:**")
            for b in results["bottlenecks"][:2]:
                lines.append(f"- **{b['bottleneck']}** ({b['technology']})")
                lines.append(f"  {b['description'][:200]}")
                lines.append(f"  Emerging solutions: {b['solutions_in_progress'][:150]}")

        return "\n".join(lines)

    @staticmethod
    def query_needs_failure_check(query: str) -> bool:
        """Heuristic: does this query need failure case cross-checking?"""
        q = query.lower()
        triggers = [
            r'安全|safety|risk|风险|adverse|不良',
            r'失败|failure|terminate|终止|pause|暂停',
            r'副作用|side effect|toxicity|毒性',
            r'撤回|retract|争议|controversy',
            r'off.?target|脱靶',
            r'obstacle|bottleneck|瓶颈|障碍|limitation|局限',
            r'为什么.*停|why.*stop|why.*fail',
        ]
        return any(re.search(pat, q, re.I) for pat in triggers)
