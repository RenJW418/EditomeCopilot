#!/usr/bin/env python3
"""
EditomeCopilot — Comprehensive Gene-Editing Literature Database Builder
========================================================================

This is the **single unified entry point** for building the most complete
gene-editing knowledge base possible.  It supersedes the older
build_full_knowledge_base.py, build_knowledge_base.py, and
enrich_knowledge_base.py scripts.

Architecture
------------
  10 conceptual Tiers × 4 data sources → DOI/Title dedup → JSON DB
  → (optional) process_knowledge_base → FAISS+BM25 index
  → (optional) ner_kg_builder → Knowledge Graph

Data sources
  1. Europe PMC   — primary, best for sorting/filtering, cursor pagination
  2. PubMed NCBI  — gold standard metadata, XML batch via WebEnv
  3. Semantic Scholar — DOI-level enrichment, citation counts, open access PDFs
  4. ClinicalTrials.gov — clinical translation evidence

Usage
-----
  # Full build (all tiers + index + KG) — takes several hours
  python scripts/build_literature_db.py --full

  # Metadata only (fast, ~30 min)
  python scripts/build_literature_db.py

  # Limit per tier for testing
  python scripts/build_literature_db.py --tier-limit 200

  # Build index after metadata collection
  python scripts/build_literature_db.py --index-only

  # Build KG after metadata collection
  python scripts/build_literature_db.py --kg-only
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.multi_source_fetcher import MultiSourceFetcher

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_DIR = ROOT / "data" / "knowledge_base"
QUARTER = (datetime.now().month - 1) // 3 + 1
VERSION = datetime.now().strftime(f"GEA_v%Y_Q{QUARTER}")

# ===========================================================================
# Gold Standard — Landmark papers that MUST be present
# ===========================================================================
GOLD_STANDARD_DOIS: List[str] = [
    # ── Foundational CRISPR ──
    "10.1126/science.1225829",        # Jinek et al. 2012 — CRISPR-Cas9 mechanism
    "10.1126/science.1231143",        # Cong et al. 2013 — Mammalian genome editing
    "10.1126/science.1232033",        # Mali et al. 2013 — RNA-guided human genome engineering
    "10.1016/j.cell.2014.05.010",     # Doudna & Charpentier 2014 — CRISPR review
    "10.1038/nbt.2647",               # Hsu et al. 2013 — SpCas9 targeting
    "10.1038/nature14299",            # Kleinstiver et al. 2015 — High-fidelity Cas9
    "10.1038/s41586-018-0686-x",      # Rees & Liu 2018 — Base editing review

    # ── Base Editing ──
    "10.1038/nature17946",            # Komor et al. 2016 — CBE (BE3)
    "10.1038/nature24644",            # Gaudelli et al. 2017 — ABE
    "10.1038/s41587-020-0453-z",      # Richter et al. 2020 — ABE8e

    # ── Prime Editing ──
    "10.1038/s41586-019-1711-4",      # Anzalone et al. 2019 — Prime Editing
    "10.1038/s41587-021-01039-7",     # Chen et al. 2021 — PE enhancements (PEmax)
    "10.1038/s41587-022-01613-7",     # Anzalone et al. 2022 — Programmable large insertions

    # ── RNA Editing ──
    "10.1126/science.aaq0180",        # Cox et al. 2017 — REPAIR (ADAR)
    "10.1126/science.aax7063",        # Abudayyeh et al. 2019 — RESCUE
    "10.1038/s41587-019-0178-z",      # Qu et al. 2019 — Programmable RNA editing

    # ── Cas Variants ──
    "10.1016/j.cell.2015.09.038",     # Zetsche et al. 2015 — Cas12a (Cpf1)
    "10.1038/nature21059",            # Abudayyeh et al. 2017 — Cas13a RNA targeting
    "10.1126/science.aam9321",        # Gootenberg et al. 2017 — SHERLOCK
    "10.1126/science.aax9249",        # Pausch et al. 2020 — CasΦ (CasPhi)
    "10.1038/s41587-020-0491-6",      # Strecker et al. 2019 — CasX (Cas12e)
    "10.1126/science.aav7271",        # Yan et al. 2019 — Cas12b
    "10.1038/s41586-021-03886-5",     # Kannan et al. 2022 — Cas7-11
    "10.1038/s41586-023-06356-2",     # Tong et al. 2023 — TnpB
    "10.1038/s41586-024-07998-8",     # new Cas variants 2024

    # ── Delivery Systems ──
    "10.1038/nbt.3471",               # Zuris et al. 2015 — RNP delivery
    "10.1038/s41586-020-2008-9",      # Gillmore et al. 2021 — In vivo NTLA-2001 (LNP)
    "10.1038/s41587-021-00933-4",     # Musunuru et al. 2021 — PCSK9 via LNP in vivo
    "10.1038/s41467-020-17029-3",     # Wei et al. 2020 — LNP optimization
    "10.1038/nbt.4199",               # Yin et al. 2017 — LNP Cas9 mRNA in vivo
    "10.1126/science.aah5297",        # Kim et al. 2017 — AAV delivery

    # ── Off-Target / Safety ──
    "10.1038/nmeth.2812",             # Fu et al. 2013 — Off-target analysis
    "10.1038/nbt.3117",               # Tsai et al. 2015 — GUIDE-seq
    "10.1038/s41586-019-1048-8",      # Zuo et al. 2019 — CBE off-target DNA
    "10.1038/s41586-019-1161-z",      # Grünewald et al. 2019 — CBE off-target RNA
    "10.1038/s41587-019-0032-3",      # Kim et al. 2019 — Digenome-seq
    "10.1038/s41592-019-0549-5",      # Bae et al. 2014 — Cas-OFFinder

    # ── Clinical Milestones ──
    "10.1056/NEJMoa2031054",          # Frangoul et al. 2021 — CTX001 SCD
    "10.1056/NEJMoa2309149",          # Vertex 2024 — Casgevy approval
    "10.1038/s41586-021-03534-y",     # Gillmore et al. 2021 — NTLA-2001 in vivo
    "10.1056/NEJMoa2314390",          # VERVE-101 clinical results

    # ── Epigenome Editing ──
    "10.1038/nbt.3199",               # Hilton et al. 2015 — dCas9-p300 epigenome editing
    "10.1038/s41587-021-00927-2",     # Nuñez et al. 2021 — CRISPRoff
    "10.1016/j.cell.2021.01.017",     # Liu et al. 2021 — dCas9-based methylation

    # ── Transposon & Retron ──
    "10.1038/s41586-019-1323-z",      # Strecker et al. 2019 — CRISPR-associated transposase
    "10.1126/science.aax9181",        # Klompe et al. 2019 — RNA-guided DNA insertion
    "10.1038/s41589-021-00927-y",     # Schubert et al. 2021 — Retron editing

    # ── Reviews & Perspectives ──
    "10.1038/s41576-019-0128-9",      # Anzalone et al. 2020 — Genome editing review
    "10.1038/s41576-023-00586-w",     # Doudna 2020 — CRISPR's future
    "10.1126/science.add8643",        # Wang & Doudna 2023 — CRISPR technology
]


# ===========================================================================
# Tiered Query Strategy — 10 layers covering all gene editing domains
# ===========================================================================

def generate_comprehensive_queries(tier_limit: int = 0) -> List[Dict]:
    """
    Return a list of query dicts. Each dict has:
      q       : PubMed/EPMC compatible boolean query
      limit   : max results to fetch for this tier
      tag     : human-readable tier label
      source  : list of data-source names
    """
    current_year = datetime.now().year
    queries: List[Dict] = []

    # Helper: default limit or override
    def _lim(default: int) -> int:
        return tier_limit if tier_limit > 0 else default

    # ── Tier 1: Core CRISPR Technologies (High-Citation Backbone) ──────────
    queries.append({
        "q": '("CRISPR" OR "Cas9" OR "Cas12" OR "Base Editing" OR "Prime Editing" OR "Gene Editing") SORT_CITED:Y',
        "limit": _lim(15000),
        "tag": "T1_Core_HighCite",
        "source": ["EuropePMC"],
    })

    # ── Tier 2: Broad Gene Editing (PubMed) ─────────────────────────────────
    queries.append({
        "q": '("CRISPR-Cas9"[Title/Abstract] OR "gene editing"[Title/Abstract] OR '
             '"genome editing"[Title/Abstract] OR "base editing"[Title/Abstract] OR '
             '"prime editing"[Title/Abstract])',
        "limit": _lim(10000),
        "tag": "T2_Broad_PubMed",
        "source": ["PubMed"],
    })

    # ── Tier 3: Novel Cas Variants & Emerging Tools (Last 4 Years) ─────────
    queries.append({
        "q": (f'("Cas13" OR "CasPhi" OR "Cas7-11" OR "Cas12a" OR "Cas12b" OR "Cas12e" OR '
              f'"CasX" OR "Cas12f" OR "Cas12j" OR "TnpB" OR "IscB" OR "Fanzor" OR '
              f'"CasLambda" OR "Cas14" OR "Un1Cas12f1" OR "compact Cas" OR '
              f'"hypercompact" OR "type V" OR "type VI") '
              f'AND PUB_YEAR:[{current_year-4} TO {current_year}]'),
        "limit": _lim(8000),
        "tag": "T3_NovelCas",
        "source": ["EuropePMC"],
    })

    # ── Tier 4: Base Editing Deep-Dive ──────────────────────────────────────
    queries.append({
        "q": ('("base editor" OR "cytosine base editor" OR "adenine base editor" OR '
              '"ABE" OR "CBE" OR "GBE" OR "AYBE" OR "glycosylase base editor" OR '
              '"dual base editor" OR "deaminase" OR "BE3" OR "BE4" OR "ABE7" OR "ABE8" OR '
              '"DddA" OR "mitochondrial base editing")'),
        "limit": _lim(5000),
        "tag": "T4_BaseEditing",
        "source": ["PubMed", "EuropePMC"],
    })

    # ── Tier 5: Prime Editing Deep-Dive ─────────────────────────────────────
    queries.append({
        "q": ('("prime editing" OR "prime editor" OR "pegRNA" OR "PEmax" OR "PE2" OR '
              '"PE3" OR "PE4" OR "PE5" OR "twinPE" OR "GRAND editing" OR '
              '"prime editing efficiency" OR "nicking" OR "reverse transcriptase" OR '
              '"programmable insertion")'),
        "limit": _lim(3000),
        "tag": "T5_PrimeEditing",
        "source": ["PubMed", "EuropePMC"],
    })

    # ── Tier 6: RNA Editing ─────────────────────────────────────────────────
    queries.append({
        "q": ('("ADAR" OR "RNA editing" OR "A-to-I editing" OR "RESCUE" OR "REPAIR" OR '
              '"Cas13" OR "Cas13d" OR "CasRx" OR "programmable RNA" OR '
              '"RNA base editing" OR "C-to-U RNA editing" OR "APOBEC RNA")'),
        "limit": _lim(5000),
        "tag": "T6_RNAEditing",
        "source": ["PubMed", "EuropePMC"],
    })

    # ── Tier 7: Delivery Systems ────────────────────────────────────────────
    delivery_kw = (
        '"lipid nanoparticle" OR "LNP" OR "AAV" OR "adeno-associated virus" OR '
        '"viral vector" OR "RNP delivery" OR "ribonucleoprotein" OR '
        '"electroporation" OR "exosome" OR "VLP" OR "virus-like particle" OR '
        '"mRNA delivery" OR "nanoparticle" OR "cell-penetrating peptide" OR '
        '"SEND" OR "eVLP" OR "engineered VLP" OR "targeted delivery"'
    )
    gene_edit_kw = '"CRISPR" OR "Cas9" OR "gene editing" OR "base editing" OR "prime editing"'
    queries.append({
        "q": f'({delivery_kw}) AND ({gene_edit_kw})',
        "limit": _lim(10000),
        "tag": "T7_Delivery",
        "source": ["PubMed", "EuropePMC"],
    })

    # ── Tier 8: Off-Target / Safety / Specificity ──────────────────────────
    queries.append({
        "q": (f'({gene_edit_kw}) AND '
              f'("off-target" OR "specificity" OR "safety" OR "immunogenicity" OR '
              f'"toxicity" OR "genotoxicity" OR "carcinogenicity" OR '
              f'"GUIDE-seq" OR "CIRCLE-seq" OR "Digenome-seq" OR "DISCOVER-seq" OR '
              f'"CHANGE-seq" OR "BLISS" OR "VIVO" OR "high-fidelity" OR '
              f'"anti-CRISPR" OR "AcrIIA" OR "immune response")'),
        "limit": _lim(6000),
        "tag": "T8_Safety",
        "source": ["PubMed", "EuropePMC"],
    })

    # ── Tier 9: Clinical Translation & Therapeutics ────────────────────────
    queries.append({
        "q": (f'({gene_edit_kw}) AND '
              f'("clinical trial" OR "phase I" OR "phase II" OR "phase III" OR '
              f'"therapy" OR "patient" OR "in vivo" OR "ex vivo" OR '
              f'"FDA" OR "EMA" OR "IND" OR "regulatory" OR '
              f'"sickle cell" OR "beta-thalassemia" OR "TTR amyloidosis" OR '
              f'"Casgevy" OR "exa-cel" OR "CTX001" OR "EDIT-101" OR '
              f'"NTLA-2001" OR "VERVE-101" OR "BEAM-101")'),
        "limit": _lim(5000),
        "tag": "T9_Clinical",
        "source": ["PubMed", "ClinicalTrials"],
    })
    # Separate ClinicalTrials.gov query
    queries.append({
        "q": "CRISPR OR gene editing OR base editing OR prime editing",
        "limit": _lim(500),
        "tag": "T9_CT_gov",
        "source": ["ClinicalTrials"],
    })

    # ── Tier 10: Epigenome Editing / CRISPRi / CRISPRa ────────────────────
    queries.append({
        "q": ('("CRISPRi" OR "CRISPRa" OR "epigenome editing" OR "dCas9" OR '
              '"CRISPR activation" OR "CRISPR interference" OR "CRISPRoff" OR '
              '"KRAB" OR "VPR" OR "VP64" OR "p300" OR "DNMT3" OR '
              '"histone modification" OR "chromatin remodeling")'),
        "limit": _lim(5000),
        "tag": "T10_Epigenome",
        "source": ["PubMed", "EuropePMC"],
    })

    # ── Tier 11: CRISPR Diagnostics ────────────────────────────────────────
    queries.append({
        "q": ('("SHERLOCK" OR "DETECTR" OR "CRISPR diagnostic" OR '
              '"CRISPR detection" OR "Cas12 diagnostic" OR "Cas13 diagnostic" OR '
              '"HUDSON" OR "STOP" OR "lateral flow" AND CRISPR)'),
        "limit": _lim(3000),
        "tag": "T11_Diagnostics",
        "source": ["EuropePMC"],
    })

    # ── Tier 12: Gene Drives & Agriculture ─────────────────────────────────
    queries.append({
        "q": ('("gene drive" OR "CRISPR agriculture" OR "crop improvement" OR '
              '"plant genome editing" OR "livestock editing" OR '
              '"allelic drive" OR "suppression drive")'),
        "limit": _lim(3000),
        "tag": "T12_GeneDrive_Agri",
        "source": ["EuropePMC"],
    })

    # ── Tier 13: Screening & Functional Genomics ──────────────────────────
    queries.append({
        "q": ('("CRISPR screen" OR "genome-wide screen" OR "functional genomics" OR '
              '"single-cell CRISPR" OR "Perturb-seq" OR "CROP-seq" OR '
              '"CRISPR library" OR "guide RNA library" OR "loss-of-function screen")'),
        "limit": _lim(5000),
        "tag": "T13_Screens",
        "source": ["PubMed", "EuropePMC"],
    })

    # ── Tier 14: Computational / gRNA Design / AI ──────────────────────────
    queries.append({
        "q": ('("gRNA design" OR "guide RNA design" OR "CRISPR machine learning" OR '
              '"deep learning CRISPR" OR "off-target prediction" OR '
              '"CRISPR efficiency prediction" OR "Cas-OFFinder" OR "CRISPRscan" OR '
              '"CHOPCHOP" OR "DeepCRISPR" OR "CRISPR-ML" OR "ABE efficiency prediction")'),
        "limit": _lim(3000),
        "tag": "T14_Computational",
        "source": ["PubMed", "EuropePMC"],
    })

    # ── Tier 15: Transposon & Retron Editing ───────────────────────────────
    queries.append({
        "q": ('("CRISPR transposon" OR "CRISPR-associated transposase" OR '
              '"CAST" OR "ShCAST" OR "Tn7" OR "retron" OR "retron editing" OR '
              '"RNA-guided transposition" OR "programmable insertion large")'),
        "limit": _lim(2000),
        "tag": "T15_Transposon_Retron",
        "source": ["EuropePMC"],
    })

    # ── Tier 16: bioRxiv Preprints (Latest, not yet peer-reviewed) ─────────
    queries.append({
        "q": (f'("CRISPR" OR "base editing" OR "prime editing" OR "gene editing") '
              f'AND SRC:PPR AND PUB_YEAR:[{current_year-2} TO {current_year}]'),
        "limit": _lim(5000),
        "tag": "T16_Preprints",
        "source": ["EuropePMC"],
    })

    # ── Tier 17: Disease-Specific Deep Queries ─────────────────────────────
    diseases = [
        ("sickle cell disease", "hemoglobin", "HBB"),
        ("beta-thalassemia", "fetal hemoglobin", "BCL11A"),
        ("Duchenne muscular dystrophy", "dystrophin", "DMD"),
        ("cystic fibrosis", "CFTR", "lung"),
        ("hereditary angioedema", "HAE", "KLKB1"),
        ("TTR amyloidosis", "transthyretin", "NTLA-2001"),
        ("hypercholesterolemia", "PCSK9", "VERVE-101"),
        ("retinal dystrophy", "CEP290", "EDIT-101"),
        ("hemophilia", "factor VIII", "factor IX"),
        ("HIV", "CCR5", "CRISPR cure"),
        ("cancer immunotherapy", "CAR-T", "CRISPR knockout"),
        ("alpha-1 antitrypsin", "SERPINA1", "liver"),
        ("Huntington", "HTT", "allele-specific"),
    ]
    for disease, kw2, kw3 in diseases:
        queries.append({
            "q": f'("{disease}" OR "{kw2}" OR "{kw3}") AND ({gene_edit_kw})',
            "limit": _lim(1000),
            "tag": f"T17_Disease_{disease.replace(' ', '_')[:20]}",
            "source": ["PubMed"],
        })

    return queries


# ===========================================================================
# Semantic Scholar Enrichment (optional, citation + open-access PDF)
# ===========================================================================

def enrich_from_semantic_scholar(articles: List[Dict], batch_size: int = 100) -> int:
    """
    For articles with a DOI, fetch citation count and open-access PDF URL
    from Semantic Scholar Graph API (free, 100 req/s with API key).
    Returns count of enriched articles.
    """
    import requests

    api_key = os.getenv("S2_API_KEY", "")
    headers = {"User-Agent": "EditomeCopilot/2.0"}
    if api_key:
        headers["x-api-key"] = api_key

    enriched = 0
    dois = [a for a in articles if a.get("doi") and not a.get("citation_count")]

    for i in range(0, len(dois), batch_size):
        batch = dois[i:i + batch_size]
        ids_param = ",".join(f"DOI:{a['doi']}" for a in batch)

        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/batch"
            resp = requests.post(
                url,
                json={"ids": [f"DOI:{a['doi']}" for a in batch]},
                params={"fields": "citationCount,isOpenAccess,openAccessPdf,year,title"},
                headers=headers,
                timeout=30,
            )
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 5))
                print(f"  [S2] Rate-limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                continue

            results = resp.json()
            for art, s2_data in zip(batch, results):
                if s2_data is None:
                    continue
                art["citation_count"] = s2_data.get("citationCount", 0)
                if s2_data.get("openAccessPdf"):
                    art["oa_pdf_url"] = s2_data["openAccessPdf"].get("url")
                enriched += 1

            time.sleep(0.5)

        except Exception as e:
            print(f"  [S2] Batch error: {e}")

    print(f"  [Semantic Scholar] Enriched {enriched}/{len(dois)} articles with citation data.")
    return enriched


# ===========================================================================
# PubMed Metadata Enrichment (fill missing authors/journal)
# ===========================================================================

def enrich_from_pubmed(articles: List[Dict], batch_size: int = 50, max_enrich: int = 2000) -> int:
    """Fill missing author/journal/abstract from PubMed for articles with a PMID."""
    import requests
    import xml.etree.ElementTree as ET

    candidates = [
        a for a in articles
        if (a.get("id") or a.get("pmid"))
        and a.get("source") in ("PubMed", "EuropePMC")
        and (not a.get("authors") or a["authors"] in ([], ["Unknown"], "Unknown") or not a.get("journal"))
    ][:max_enrich]

    if not candidates:
        return 0

    enriched = 0
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        pmids = [str(a.get("id") or a.get("pmid")) for a in batch if str(a.get("id") or a.get("pmid")).isdigit()]
        if not pmids:
            continue

        try:
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            resp = requests.post(url, data={
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
            }, timeout=30)
            resp.raise_for_status()

            root = ET.fromstring(resp.content)
            pmid_to_data: Dict[str, Dict] = {}

            for article_xml in root.findall(".//PubmedArticle"):
                pmid = article_xml.findtext(".//PMID")
                if not pmid:
                    continue

                authors = []
                for author in article_xml.findall(".//Author"):
                    last = author.findtext("LastName")
                    fore = author.findtext("ForeName")
                    if last and fore:
                        authors.append(f"{fore} {last}")
                    elif last:
                        authors.append(last)

                journal = article_xml.findtext(".//Journal/Title")
                abstract_parts = article_xml.findall(".//AbstractText")
                abstract = " ".join("".join(elem.itertext()) for elem in abstract_parts)

                pmid_to_data[pmid] = {
                    "authors": authors,
                    "journal": journal,
                    "abstract": abstract,
                }

            for art in batch:
                pid = str(art.get("id") or art.get("pmid"))
                if pid in pmid_to_data:
                    data = pmid_to_data[pid]
                    if data.get("authors") and (not art.get("authors") or art["authors"] in ([], "Unknown")):
                        art["authors"] = data["authors"]
                    if data.get("journal") and not art.get("journal"):
                        art["journal"] = data["journal"]
                    if data.get("abstract") and (not art.get("abstract") or len(art.get("abstract", "")) < 50):
                        art["abstract"] = data["abstract"]
                    enriched += 1

            time.sleep(0.5)
        except Exception as e:
            print(f"  [PubMed Enrich] Batch error: {e}")

    print(f"  [PubMed Enrich] Updated {enriched}/{len(candidates)} articles.")
    return enriched


# ===========================================================================
# Knowledge Base Builder
# ===========================================================================

class ComprehensiveLiteratureBuilder:
    """
    Unified builder for the gene-editing knowledge base.
    
    Produces:
      data/knowledge_base/literature_db_GEA_v{YEAR}_Q{Q}.json
      data/knowledge_base/report_GEA_v{YEAR}_Q{Q}.md
    """

    def __init__(self, tier_limit: int = 0, skip_enrich: bool = False):
        self.output_dir = OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.fetcher = MultiSourceFetcher()
        self.database: Dict[str, Dict] = {}  # gea_id → article
        self.seen_dois: set = set()
        self.seen_titles: set = set()
        self.tier_limit = tier_limit
        self.skip_enrich = skip_enrich
        self.version = VERSION
        self.stats: Dict[str, Any] = {
            "tiers": {},
            "sources": {"PubMed": 0, "EuropePMC": 0, "ClinicalTrials": 0},
        }

    # ── Persistence ────────────────────────────────────────────────────────

    def _find_latest_db(self) -> Optional[Path]:
        files = sorted(self.output_dir.glob("literature_db_GEA_v*.json"))
        return files[-1] if files else None

    def load_existing(self):
        latest = self._find_latest_db()
        if not latest or not latest.exists():
            print("No existing database found. Starting fresh.")
            return
        print(f"Loading existing database: {latest}")
        try:
            with open(latest, "r", encoding="utf-8") as f:
                data = json.load(f)
            for art in data:
                gea_id = art.get("gea_id")
                if not gea_id:
                    continue
                self.database[gea_id] = art
                doi = art.get("doi")
                if doi:
                    self.seen_dois.add(doi.lower().strip())
                title = art.get("title")
                if title:
                    self.seen_titles.add(self._normalize_title(title))
            print(f"  Loaded {len(self.database)} existing records.")
        except Exception as e:
            print(f"  Error loading: {e}")

    @staticmethod
    def _normalize_title(title: str) -> str:
        return "".join(c for c in title.lower() if c.isalnum())

    def save(self):
        filepath = self.output_dir / f"literature_db_{self.version}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(list(self.database.values()), f, indent=1, ensure_ascii=False)
        print(f"Saved {len(self.database)} records → {filepath}")
        return filepath

    # ── Deduplication ──────────────────────────────────────────────────────

    def _is_duplicate(self, art: Dict) -> bool:
        doi = art.get("doi")
        if doi and doi.lower().strip() in self.seen_dois:
            return True
        title = art.get("title")
        if title and self._normalize_title(title) in self.seen_titles:
            return True
        return False

    def _register(self, art: Dict):
        doi = art.get("doi")
        if doi:
            self.seen_dois.add(doi.lower().strip())
        title = art.get("title")
        if title:
            self.seen_titles.add(self._normalize_title(title))

    def add_article(self, art: Dict) -> bool:
        """Add article if not duplicate. Returns True if added."""
        if self._is_duplicate(art):
            return False
        # Generate GEA ID
        gea_id = art.get("gea_id")
        if not gea_id:
            if art.get("source") == "PubMed" and art.get("id"):
                gea_id = f"PMID:{art['id']}"
            elif art.get("doi"):
                gea_id = f"DOI:{art['doi']}"
            elif art.get("source") == "ClinicalTrials.gov" and art.get("id"):
                gea_id = f"NCT:{art['id']}"
            else:
                gea_id = "HASH:" + hashlib.md5(art.get("title", "").encode()).hexdigest()[:12]
            art["gea_id"] = gea_id

        if gea_id in self.database:
            # Merge: prefer non-empty fields from new article
            existing = self.database[gea_id]
            for k, v in art.items():
                if v and (not existing.get(k) or existing[k] in ("", "Unknown", [], None)):
                    existing[k] = v
            self._register(existing)
            return False

        art.setdefault("fetched_at", datetime.now().isoformat())
        self.database[gea_id] = art
        self._register(art)
        return True

    # ── Gold Standard Ingestion ────────────────────────────────────────────

    def ingest_gold_standard(self):
        print(f"\n{'='*60}")
        print(f"Ingesting Gold Standard Papers ({len(GOLD_STANDARD_DOIS)} DOIs)")
        print(f"{'='*60}")
        added = 0
        for doi in GOLD_STANDARD_DOIS:
            # Check if already in DB
            if doi.lower().strip() in self.seen_dois:
                continue
            try:
                results = self.fetcher.fetch_europe_pmc(f'DOI:"{doi}"', max_results=1)
                if results and self.add_article(results[0]):
                    added += 1
                    print(f"  + {doi}")
                elif not results:
                    print(f"  ? Not found: {doi}")
                time.sleep(0.3)
            except Exception as e:
                print(f"  ! Error: {doi} — {e}")
        print(f"  Gold Standard: added {added} new papers.")

    # ── Tier-based Fetching ────────────────────────────────────────────────

    def fetch_all_tiers(self):
        queries = generate_comprehensive_queries(self.tier_limit)

        source_map = {
            "PubMed": self.fetcher.fetch_pubmed,
            "EuropePMC": self.fetcher.fetch_europe_pmc,
            "ClinicalTrials": self.fetcher.fetch_clinical_trials,
        }

        total_fetched = 0
        total_added = 0

        for idx, tier in enumerate(queries):
            q = tier["q"]
            limit = tier["limit"]
            tag = tier["tag"]
            sources = tier["source"]

            print(f"\n{'─'*60}")
            print(f"[{idx+1}/{len(queries)}] {tag}")
            print(f"  Query: {q[:100]}{'...' if len(q) > 100 else ''}")
            print(f"  Limit: {limit} | Sources: {sources}")
            print(f"{'─'*60}")

            tier_fetched = 0
            tier_added = 0

            for src in sources:
                fn = source_map.get(src)
                if not fn:
                    print(f"  ! Unknown source: {src}")
                    continue

                try:
                    time.sleep(1)  # Respectful delay between sources
                    results = fn(q, max_results=limit)
                    print(f"  [{src}] Returned {len(results)} results")

                    batch_added = 0
                    for art in results:
                        art["tier_tag"] = tag
                        if self.add_article(art):
                            batch_added += 1
                            total_added += 1

                    tier_fetched += len(results)
                    tier_added += batch_added
                    self.stats["sources"][src] = self.stats["sources"].get(src, 0) + len(results)

                    # Auto-save every 500 new articles
                    if total_added > 0 and total_added % 500 == 0:
                        print(f"  [Auto-save] {total_added} new articles so far...")
                        self.save()

                except Exception as e:
                    print(f"  ! [{src}] Error: {e}")

            total_fetched += tier_fetched
            self.stats["tiers"][tag] = {"fetched": tier_fetched, "added": tier_added}
            print(f"  → Tier result: fetched={tier_fetched}, new={tier_added}, total_db={len(self.database)}")

            # Save after each tier
            self.save()

        return total_fetched, total_added

    # ── Coverage Validation ────────────────────────────────────────────────

    def validate_coverage(self) -> Dict:
        found = 0
        missing = []
        for doi in GOLD_STANDARD_DOIS:
            if doi.lower().strip() in self.seen_dois:
                found += 1
            else:
                missing.append(doi)

        pct = (found / len(GOLD_STANDARD_DOIS)) * 100 if GOLD_STANDARD_DOIS else 0
        print(f"\n  Gold Standard Coverage: {pct:.1f}% ({found}/{len(GOLD_STANDARD_DOIS)})")
        if missing:
            print(f"  Missing DOIs: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        return {"coverage_pct": pct, "found": found, "total": len(GOLD_STANDARD_DOIS), "missing": missing}

    # ── Report Generation ──────────────────────────────────────────────────

    def generate_report(self, initial_count: int, total_fetched: int, total_added: int, coverage: Dict):
        lines = [
            f"# Gene Editing Knowledge Base — Build Report",
            f"",
            f"**Version**: `{self.version}`  ",
            f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Builder**: `scripts/build_literature_db.py`",
            f"",
            f"## Summary",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Initial DB size | {initial_count:,} |",
            f"| Records fetched this run | {total_fetched:,} |",
            f"| New records added | {total_added:,} |",
            f"| **Final DB size** | **{len(self.database):,}** |",
            f"| Gold Standard coverage | {coverage['coverage_pct']:.1f}% ({coverage['found']}/{coverage['total']}) |",
            f"",
            f"## Source Breakdown",
            f"| Source | Records Fetched |",
            f"|--------|----------------|",
        ]
        for src, count in sorted(self.stats["sources"].items()):
            lines.append(f"| {src} | {count:,} |")

        lines += [
            f"",
            f"## Tier Breakdown",
            f"| Tier | Fetched | Added |",
            f"|------|---------|-------|",
        ]
        for tag, data in self.stats["tiers"].items():
            lines.append(f"| {tag} | {data['fetched']:,} | {data['added']:,} |")

        if coverage.get("missing"):
            lines += [
                f"",
                f"## Missing Gold Standard Papers",
            ]
            for doi in coverage["missing"]:
                lines.append(f"- `{doi}`")

        lines += [
            f"",
            f"## Quality Notes",
            f"- Deduplication: DOI (normalized) + Title hash",
            f"- Sources: PubMed NCBI, Europe PMC, ClinicalTrials.gov",
            f"- Enrichment: Semantic Scholar citation counts, PubMed metadata",
        ]

        report_path = self.output_dir / f"report_{self.version}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  Report saved → {report_path}")

    # ── Main Run ───────────────────────────────────────────────────────────

    def run(self):
        print(f"\n{'='*60}")
        print(f"  EditomeCopilot Literature DB Builder")
        print(f"  Version: {self.version}")
        print(f"  Tier limit: {'ALL' if self.tier_limit == 0 else self.tier_limit}")
        print(f"{'='*60}\n")

        # 1. Load existing DB
        self.load_existing()
        initial_count = len(self.database)

        # 2. Gold standard
        self.ingest_gold_standard()

        # 3. Tiered fetching
        total_fetched, total_added = self.fetch_all_tiers()

        # 4. Enrichment
        if not self.skip_enrich:
            print(f"\n{'='*60}")
            print("Post-processing: Enrichment")
            print(f"{'='*60}")
            articles = list(self.database.values())
            enrich_from_pubmed(articles, max_enrich=2000)
            enrich_from_semantic_scholar(articles)
            self.save()

        # 5. Validate
        coverage = self.validate_coverage()

        # 6. Report
        self.generate_report(initial_count, total_fetched, total_added, coverage)

        print(f"\n{'='*60}")
        print(f"  BUILD COMPLETE")
        print(f"  Final database: {len(self.database):,} records")
        print(f"  Gold coverage: {coverage['coverage_pct']:.1f}%")
        print(f"{'='*60}")

        return self.save()


# ===========================================================================
# Post-build: FAISS Index
# ===========================================================================

def build_faiss_index(db_path: Path):
    """Build FAISS + BM25 index from the literature JSON database."""
    print(f"\n{'='*60}")
    print("Building FAISS + BM25 Index")
    print(f"{'='*60}")

    # Import here to avoid loading heavy models during metadata-only runs
    sys.path.insert(0, str(ROOT))
    from scripts.process_knowledge_base import KnowledgeBaseProcessor

    processor = KnowledgeBaseProcessor(str(db_path))
    processor.run()


# ===========================================================================
# Post-build: Knowledge Graph
# ===========================================================================

def build_knowledge_graph(db_path: Path, pubtator: bool = False, limit: int = 0):
    """Build the NER-based knowledge graph from the literature JSON database."""
    print(f"\n{'='*60}")
    print("Building Knowledge Graph via NER")
    print(f"{'='*60}")

    from scripts.ner_kg_builder import run as run_kg_builder
    # Override the KB_PATH used by ner_kg_builder
    import scripts.ner_kg_builder as nkb
    nkb.KB_PATH = db_path
    run_kg_builder(limit=limit, pubtator_mode=pubtator)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EditomeCopilot: Comprehensive Gene-Editing Literature DB Builder"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full pipeline: fetch → enrich → FAISS index → KG build",
    )
    parser.add_argument(
        "--tier-limit",
        type=int,
        default=0,
        help="Max results per tier (0 = use tier defaults). Good for testing.",
    )
    parser.add_argument(
        "--skip-enrich",
        action="store_true",
        help="Skip Semantic Scholar / PubMed enrichment step.",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only build FAISS index from existing literature DB JSON.",
    )
    parser.add_argument(
        "--kg-only",
        action="store_true",
        help="Only build Knowledge Graph from existing literature DB JSON.",
    )
    parser.add_argument(
        "--pubtator",
        action="store_true",
        help="Use PubTator API for KG NER (instead of local scispaCy).",
    )
    parser.add_argument(
        "--kg-limit",
        type=int,
        default=0,
        help="Max articles to process for KG (0 = all).",
    )

    args = parser.parse_args()

    # ── Index-only mode ──
    if args.index_only:
        db_path = sorted(OUTPUT_DIR.glob("literature_db_GEA_v*.json"))
        if not db_path:
            print("ERROR: No literature DB found. Run metadata collection first.")
            sys.exit(1)
        build_faiss_index(db_path[-1])
        return

    # ── KG-only mode ──
    if args.kg_only:
        db_path = sorted(OUTPUT_DIR.glob("literature_db_GEA_v*.json"))
        if not db_path:
            print("ERROR: No literature DB found. Run metadata collection first.")
            sys.exit(1)
        build_knowledge_graph(db_path[-1], pubtator=args.pubtator, limit=args.kg_limit)
        return

    # ── Full or metadata-only build ──
    builder = ComprehensiveLiteratureBuilder(
        tier_limit=args.tier_limit,
        skip_enrich=args.skip_enrich,
    )
    db_path = builder.run()

    if args.full:
        build_faiss_index(db_path)
        build_knowledge_graph(db_path, pubtator=args.pubtator, limit=args.kg_limit)


if __name__ == "__main__":
    main()
