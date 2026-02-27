"""
Patent / IP Landscape Module
=============================
Core patents, holders, licensing status, and IP risk assessment
for each gene editing technology.

Purpose: Annotate editing strategy recommendations with IP context
so users are aware of licensing requirements and freedom-to-operate risks.

Data source: Public patent databases (USPTO, EPO), Broad/UC Berkeley licensing
announcements, published IP analyses, press releases.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Core Patent Database — Key Gene Editing Patents
# ─────────────────────────────────────────────────────────────────────────────
CORE_PATENTS: List[Dict[str, Any]] = [
    # ── CRISPR-Cas9 Foundational ────────────────────────────────────────
    {
        "patent_id": "US10,266,850",
        "title": "CRISPR-Cas9 in eukaryotic cells",
        "holder": "Broad Institute / MIT (Feng Zhang)",
        "technology": "CRISPR-Cas9",
        "filing_date": "2012-12-12",
        "status": "Granted (upheld in interference)",
        "scope": "Use of CRISPR-Cas9 in eukaryotic cells (mammalian, plant)",
        "licensing": "Licensed via ERS Genomics (Broad) for research. Therapeutics: case-by-case negotiation. Broad holds eukaryotic rights in US.",
        "territory": "US",
        "importance": "FOUNDATIONAL — Required for any SpCas9 therapeutic in US",
        "note": "Won USPTO interference proceeding vs UC Berkeley (2022). Berkeley has separate composition patents.",
    },
    {
        "patent_id": "EP2,771,468",
        "title": "CRISPR-Cas9 composition of matter",
        "holder": "UC Berkeley / University of Vienna (Doudna, Charpentier)",
        "technology": "CRISPR-Cas9",
        "filing_date": "2012-06-29 (priority: 2012-05-25)",
        "status": "Granted in EP; limited in US",
        "scope": "Composition of matter for CRISPR-Cas9 system (not limited to eukaryotic cells)",
        "licensing": "Licensed via Intellia Therapeutics (exclusive therapeutics), Caribou Biosciences. CRISPR Therapeutics has separate license from Charpentier.",
        "territory": "EP, JP, CN, KR (strong); US (limited to non-eukaryotic)",
        "importance": "FOUNDATIONAL — Key European patent. Critical for any EU clinical development.",
        "note": "Berkeley composition patent is broader but Broad has eukaryotic use patent in US. Both needed for full FTO.",
    },
    {
        "patent_id": "US10,930,367",
        "title": "Anti-CRISPR proteins",
        "holder": "UC San Francisco (Joseph Bondy-Denomy)",
        "technology": "CRISPR-Cas9 (control/safety)",
        "filing_date": "2017-06-05",
        "status": "Granted",
        "scope": "Anti-CRISPR proteins (AcrIIA4, AcrIIA2) for controlling Cas9 activity",
        "licensing": "Available for licensing",
        "territory": "US, EP",
        "importance": "MODERATE — Useful for safety switches in gene editing therapies",
        "note": "Growing importance as regulatory agencies consider 'off-switch' requirements.",
    },
    # ── Base Editing ────────────────────────────────────────────────────
    {
        "patent_id": "US10,167,457",
        "title": "Cytosine Base Editors (CBE)",
        "holder": "Harvard / Broad Institute (David Liu)",
        "technology": "CBE",
        "filing_date": "2015-04-06",
        "status": "Granted",
        "scope": "Fusion of Cas9 nickase + cytidine deaminase (APOBEC) + UGI for C→T editing",
        "licensing": "Licensed exclusively to Beam Therapeutics for therapeutics. Research use via Broad licensing.",
        "territory": "US, EP, JP, CN",
        "importance": "FOUNDATIONAL — Required for any CBE therapeutic",
        "note": "Beam Therapeutics founded by David Liu; exclusive therapeutic licensee.",
    },
    {
        "patent_id": "US10,113,163",
        "title": "Adenine Base Editors (ABE)",
        "holder": "Harvard / Broad Institute (David Liu)",
        "technology": "ABE",
        "filing_date": "2017-10-25",
        "status": "Granted",
        "scope": "Evolved TadA adenosine deaminase fused to Cas9 nickase for A→G editing",
        "licensing": "Licensed exclusively to Beam Therapeutics for therapeutics.",
        "territory": "US, EP, JP, CN",
        "importance": "FOUNDATIONAL — Required for any ABE therapeutic. VERVE-101 uses license from Beam.",
        "note": "VERVE-101 (PCSK9) operates under sublicense from Beam.",
    },
    # ── Prime Editing ───────────────────────────────────────────────────
    {
        "patent_id": "US11,447,770",
        "title": "Prime Editing",
        "holder": "Harvard / Broad Institute (David Liu, Andrew Anzalone)",
        "technology": "Prime Editing",
        "filing_date": "2019-10-21",
        "status": "Granted",
        "scope": "Cas9 nickase fused to RT + pegRNA for search-and-replace editing",
        "licensing": "Licensed exclusively to Prime Medicine for therapeutics.",
        "territory": "US, EP, JP, CN",
        "importance": "FOUNDATIONAL — Required for any PE therapeutic",
        "note": "Prime Medicine is the exclusive therapeutic licensee. Very broad claims.",
    },
    # ── Cas12 / Cpf1 ───────────────────────────────────────────────────
    {
        "patent_id": "US9,790,490",
        "title": "CRISPR-Cpf1 (Cas12a)",
        "holder": "Broad Institute / MIT (Feng Zhang)",
        "technology": "Cas12a",
        "filing_date": "2015-06-18",
        "status": "Granted",
        "scope": "Cas12a (Cpf1) nuclease for genome editing — distinct PAM (TTTV), staggered cut",
        "licensing": "Licensed to Editas Medicine (primary). Research: via Broad.",
        "territory": "US, EP",
        "importance": "FOUNDATIONAL — Required for Cas12a therapeutics (EDIT-301, etc.)",
        "note": "Cas12a has advantages over Cas9 for some applications (T-rich PAM, self-processing crRNA array).",
    },
    # ── Delivery ─────────────────────────────────────────────────────────
    {
        "patent_id": "US10,898,574",
        "title": "Ionizable LNP for nucleic acid delivery",
        "holder": "Arbutus Biopharma / Genevant Sciences",
        "technology": "LNP delivery",
        "filing_date": "2014-09-05",
        "status": "Granted (partially challenged)",
        "scope": "Ionizable lipid MC3 and related formulations for LNP delivery of nucleic acids",
        "licensing": "Complex: Genevant (Roivant/Arbutus JV) licenses to multiple. Alnylam has separate license. Moderna settled. Gene editing companies negotiate individually.",
        "territory": "US, EP",
        "importance": "HIGH — LNP delivery is the primary in vivo delivery method for mRNA-based editing",
        "note": "Multiple LNP lipid patents from Moderna, Acuitas, Arbutus overlap. FTO analysis required for any LNP-delivered therapeutic.",
    },
    {
        "patent_id": "US11,174,500",
        "title": "Engineered AAV capsids (MyoAAV, PHP.eB)",
        "holder": "Various (Broad, Caltech, Harvard)",
        "technology": "AAV delivery",
        "filing_date": "2018-2020",
        "status": "Multiple grants",
        "scope": "Tissue-specific engineered AAV capsids for improved tropism",
        "licensing": "Fragmented. Each capsid variant has different IP holder. MyoAAV → Broad. PHP.eB → Caltech/Voyager.",
        "territory": "US, EP",
        "importance": "HIGH for tissue-specific in vivo delivery",
        "note": "AAV capsid IP landscape is very fragmented. New capsids being engineered (AI-directed evolution) may circumvent existing patents.",
    },
    # ── RNA Editing ──────────────────────────────────────────────────────
    {
        "patent_id": "Multiple WO/2020/xxx",
        "title": "ADAR-recruiting oligonucleotides for RNA editing",
        "holder": "Wave Life Sciences, ProQR Therapeutics",
        "technology": "RNA editing (ADAR)",
        "filing_date": "2019-2021",
        "status": "Granted / Pending",
        "scope": "Chemically modified ASOs that recruit endogenous ADAR for A→I(G) RNA editing",
        "licensing": "Wave: WVE-006 (AATD), WVE-003 (HD). ProQR: Axiomer platform.",
        "territory": "US, EP, JP",
        "importance": "MODERATE-HIGH — Growing field. No DNA modification → different regulatory path.",
        "note": "RNA editing avoids DNA modification IP (separate from CRISPR families). May be strategically important for reversible applications.",
    },
    # ── CRISPRi / CRISPRa ──────────────────────────────────────────────
    {
        "patent_id": "US10,190,137",
        "title": "CRISPR interference (CRISPRi) and activation (CRISPRa)",
        "holder": "UC San Francisco / UC Berkeley (Jonathan Weissman, Stanley Qi, Doudna)",
        "technology": "CRISPRi/CRISPRa",
        "filing_date": "2013-03-15",
        "status": "Granted",
        "scope": "dCas9 fused to transcriptional repressors (KRAB) or activators (VP64, p65, Rta) for gene regulation",
        "licensing": "Available for licensing (academic + commercial).",
        "territory": "US, EP",
        "importance": "MODERATE — Important for epigenome editing, gene regulation without DNA breaks",
        "note": "Growing interest in CRISPRi for 'reversible gene therapy' — especially for gain-of-function diseases.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Technology → IP Summary
# ─────────────────────────────────────────────────────────────────────────────
TECHNOLOGY_IP_SUMMARY: Dict[str, Dict[str, str]] = {
    "CRISPR-Cas9": {
        "key_holders": "Broad Institute (eukaryotic use, US); UC Berkeley/Vienna (composition, EP/global)",
        "licensees": "Editas Medicine (Broad), Intellia Therapeutics (Berkeley/Caribou), CRISPR Therapeutics (Charpentier)",
        "fto_risk": "HIGH — Dual patent families required for full FTO. US and EP have different holders.",
        "trend": "Many foundational patents expire 2032-2035. New IP focuses on improved Cas9 variants, delivery, and applications.",
    },
    "ABE": {
        "key_holders": "Harvard/Broad (David Liu)",
        "licensees": "Beam Therapeutics (exclusive); Verve Therapeutics (sublicense for PCSK9)",
        "fto_risk": "HIGH — Beam has exclusive therapeutic license. Must sublicense from Beam.",
        "trend": "Next-gen ABE variants (ABE8e, ABE9) may have additional IP. Beam actively filing continuation patents.",
    },
    "CBE": {
        "key_holders": "Harvard/Broad (David Liu)",
        "licensees": "Beam Therapeutics (exclusive)",
        "fto_risk": "HIGH — Similar to ABE. Beam controls.",
        "trend": "Alternative deaminase domains (non-APOBEC) being explored to design around Liu patents.",
    },
    "Prime Editing": {
        "key_holders": "Harvard/Broad (David Liu)",
        "licensees": "Prime Medicine (exclusive)",
        "fto_risk": "VERY HIGH — Broadest claims. Prime Medicine controls all therapeutic use.",
        "trend": "PE is still early. Patent landscape will become clearer as technology matures.",
    },
    "Cas12a": {
        "key_holders": "Broad Institute (Feng Zhang)",
        "licensees": "Editas Medicine",
        "fto_risk": "MODERATE-HIGH — Fewer overlapping patents than Cas9.",
        "trend": "Alternative Cas12 variants (Cas12b, Cas12f/Cas14 for mini-CRISPR) have different IP holders.",
    },
    "RNA editing (ADAR)": {
        "key_holders": "Wave Life Sciences, ProQR/Eli Lilly",
        "licensees": "Platform-specific (Wave: stereopure ASO; ProQR: Axiomer)",
        "fto_risk": "MODERATE — Newer field, IP less consolidated. No CRISPR overlap.",
        "trend": "Rapidly expanding. Potential to avoid DNA editing IP entirely. ADAR recruiting chemistry is key IP battleground.",
    },
    "LNP delivery": {
        "key_holders": "Arbutus/Genevant, Acuitas, Moderna/Sanofi",
        "licensees": "Multiple cross-licenses (COVID vaccine settlements shifted landscape)",
        "fto_risk": "HIGH — Complex overlapping patents on ionizable lipids. Need FTO analysis per lipid chemistry.",
        "trend": "Post-COVID, LNP IP landscape is clearer but expensive. New lipid chemistries being developed to design around.",
    },
}


class PatentLandscape:
    """Gene editing patent and IP landscape intelligence."""

    def __init__(self):
        self.patents = CORE_PATENTS
        self.ip_summary = TECHNOLOGY_IP_SUMMARY

    def search(self, technology: str = None, query: str = None) -> Dict[str, Any]:
        """
        Search patent database by technology or keywords.

        Returns:
            Dict with matching patents and IP summary
        """
        results = {
            "patents": [],
            "ip_summary": None,
        }

        # Technology filter
        if technology:
            tech_lower = technology.lower()
            for p in self.patents:
                if tech_lower in p.get("technology", "").lower():
                    results["patents"].append(p)

            # IP summary
            for tech_key, summary in self.ip_summary.items():
                if tech_lower in tech_key.lower() or any(
                    t in tech_lower for t in tech_key.lower().split("/")):
                    results["ip_summary"] = summary
                    results["ip_summary"]["technology"] = tech_key
                    break

        # Keyword search
        if query:
            q = query.lower()
            for p in self.patents:
                text = " ".join(str(v) for v in p.values()).lower()
                if any(w in text for w in q.split() if len(w) > 2):
                    if p not in results["patents"]:
                        results["patents"].append(p)

        return results

    def get_ip_risk(self, technology: str) -> str:
        """Return IP risk level for a technology."""
        tech_lower = technology.lower()
        for tech_key, summary in self.ip_summary.items():
            if tech_lower in tech_key.lower() or any(
                t in tech_lower for t in tech_key.lower().split("/")):
                return summary.get("fto_risk", "UNKNOWN")
        return "UNKNOWN"

    def format_context(self, technology: str) -> str:
        """Format IP landscape as structured text for LLM context injection."""
        results = self.search(technology=technology)

        if not results["patents"] and not results["ip_summary"]:
            return ""

        lines = ["### 📋 Patent / IP Landscape"]

        ip = results.get("ip_summary")
        if ip:
            lines.append(f"**Technology:** {ip.get('technology', technology)}")
            lines.append(f"**Key holders:** {ip.get('key_holders', 'N/A')}")
            lines.append(f"**Exclusive licensees:** {ip.get('licensees', 'N/A')}")
            lines.append(f"**FTO risk:** {ip.get('fto_risk', 'N/A')}")
            lines.append(f"**Trend:** {ip.get('trend', 'N/A')}")

        patents = results.get("patents", [])
        if patents:
            lines.append(f"\n**Key Patents ({len(patents)}):**")
            for p in patents[:3]:
                lines.append(f"- **{p['patent_id']}** — {p['title']}")
                lines.append(f"  Holder: {p['holder']} | Status: {p['status']}")
                lines.append(f"  Scope: {p['scope'][:150]}")
                lines.append(f"  Licensing: {p['licensing'][:150]}")

        return "\n".join(lines) if len(lines) > 1 else ""

    @staticmethod
    def query_needs_patent(query: str) -> bool:
        """Heuristic: does this query relate to patents or IP?"""
        q = query.lower()
        triggers = [
            r'patent|专利|ip |知识产权',
            r'licens|许可|授权',
            r'freedom.?to.?operat|fto',
            r'谁.*拥有|who.*own|holder',
            r'商业化|commerciali|industrial',
        ]
        return any(re.search(pat, q, re.I) for pat in triggers)
