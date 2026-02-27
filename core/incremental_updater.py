"""
Incremental Updater — Automated Literature & Knowledge Base Updates
====================================================================
Inspired by: Continuous RAG update best practices.

Purpose:
1. Periodically (weekly/monthly) check PubMed for new gene-editing papers
2. Deduplicate against existing literature DB
3. Tag new documents (U-Retrieval tags)
4. Incrementally update FAISS index (no full rebuild)
5. Update Triple Graph with new entities
6. Generate update report
"""

from __future__ import annotations

import json
import os
import hashlib
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import requests
except ImportError:
    requests = None

try:
    import faiss
except ImportError:
    faiss = None


class IncrementalUpdater:
    """Automated incremental updater for literature KB."""

    # Core PubMed search queries for gene editing (compact version of build_literature_db.py tiers)
    WATCH_QUERIES: List[str] = [
        "CRISPR gene editing",
        "base editing therapy",
        "prime editing",
        "RNA editing ADAR therapeutic",
        "gene therapy clinical trial",
        "in vivo genome editing LNP",
        "off-target CRISPR safety",
        "CAR-T CRISPR allogeneic",
        "epigenome editing CRISPRi CRISPRa",
    ]

    def __init__(
        self,
        literature_db_path: Optional[str] = None,
        faiss_db_dir: Optional[str] = None,
        update_log_path: Optional[str] = None,
    ):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.literature_db_path = literature_db_path or os.path.join(
            base_dir, "data", "knowledge_base", "literature_db_GEA_v2026_Q1.json"
        )
        self.faiss_db_dir = faiss_db_dir or os.path.join(base_dir, "data", "faiss_db")
        self.update_log_path = update_log_path or os.path.join(
            base_dir, "data", "knowledge_base", "update_log.json"
        )

        # Load existing PMIDs for dedup
        self.existing_pmids: Set[str] = set()
        self.existing_dois: Set[str] = set()
        self._load_existing_ids()

    def _load_existing_ids(self):
        """Load existing PMIDs and DOIs from literature DB."""
        if not os.path.exists(self.literature_db_path):
            return
        try:
            with open(self.literature_db_path, "r", encoding="utf-8") as f:
                db = json.load(f)
            for rec in db:
                if rec.get("pmid"):
                    self.existing_pmids.add(str(rec["pmid"]))
                if rec.get("doi"):
                    self.existing_dois.add(rec["doi"].lower())
            print(f"[IncrementalUpdater] Loaded {len(self.existing_pmids)} existing PMIDs, {len(self.existing_dois)} DOIs")
        except Exception as e:
            print(f"[IncrementalUpdater] Error loading existing DB: {e}")

    # ─── Step 1: Fetch New Papers from PubMed ─────────────────────────────

    def fetch_new_from_pubmed(
        self,
        days_back: int = 7,
        max_per_query: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Search PubMed for papers published in the last N days.
        Deduplicate against existing DB.

        Returns list of new paper records.
        """
        if not requests:
            print("[IncrementalUpdater] requests library not available")
            return []

        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y/%m/%d")
        to_date = datetime.now().strftime("%Y/%m/%d")

        new_papers = []
        seen_pmids: Set[str] = set()

        for query in self.WATCH_QUERIES:
            try:
                # ESearch
                search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                params = {
                    "db": "pubmed",
                    "term": f"{query} AND ({from_date}[PDAT] : {to_date}[PDAT])",
                    "retmax": max_per_query,
                    "retmode": "json",
                }
                resp = requests.get(search_url, params=params, timeout=30)
                data = resp.json()
                pmids = data.get("esearchresult", {}).get("idlist", [])

                # Filter out existing
                new_pmids = [
                    p for p in pmids
                    if p not in self.existing_pmids and p not in seen_pmids
                ]
                seen_pmids.update(new_pmids)

                if not new_pmids:
                    continue

                # EFetch details
                fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                # Process in batches of 50
                for i in range(0, len(new_pmids), 50):
                    batch = new_pmids[i:i+50]
                    fetch_params = {
                        "db": "pubmed",
                        "id": ",".join(batch),
                        "rettype": "xml",
                        "retmode": "xml",
                    }
                    fetch_resp = requests.get(fetch_url, params=fetch_params, timeout=60)

                    # Parse XML (lightweight, avoiding lxml dependency)
                    papers = self._parse_pubmed_xml(fetch_resp.text)
                    new_papers.extend(papers)

                    time.sleep(0.5)  # Rate limit

            except Exception as e:
                print(f"[IncrementalUpdater] Error fetching '{query}': {e}")
                continue

        print(f"[IncrementalUpdater] Found {len(new_papers)} new papers (last {days_back} days)")
        return new_papers

    def _parse_pubmed_xml(self, xml_text: str) -> List[Dict[str, Any]]:
        """Lightweight PubMed XML parser (no lxml needed)."""
        import re
        papers = []

        # Split by article
        articles = re.findall(r'<PubmedArticle>(.*?)</PubmedArticle>', xml_text, re.DOTALL)
        for article in articles:
            try:
                pmid_m = re.search(r'<PMID[^>]*>(\d+)</PMID>', article)
                title_m = re.search(r'<ArticleTitle>(.*?)</ArticleTitle>', article, re.DOTALL)
                abstract_m = re.search(r'<AbstractText[^>]*>(.*?)</AbstractText>', article, re.DOTALL)
                doi_m = re.search(r'<ArticleId IdType="doi">(.*?)</ArticleId>', article)
                year_m = re.search(r'<PubDate>.*?<Year>(\d{4})</Year>', article, re.DOTALL)
                journal_m = re.search(r'<Title>(.*?)</Title>', article)

                if not pmid_m:
                    continue

                # Clean HTML tags
                def clean(text):
                    if not text:
                        return ""
                    return re.sub(r'<[^>]+>', '', text).strip()

                record = {
                    "pmid": pmid_m.group(1),
                    "title": clean(title_m.group(1) if title_m else ""),
                    "abstract": clean(abstract_m.group(1) if abstract_m else ""),
                    "doi": doi_m.group(1) if doi_m else "",
                    "pub_year": year_m.group(1) if year_m else "",
                    "journal": clean(journal_m.group(1) if journal_m else ""),
                    "source": "pubmed_incremental",
                    "fetched_at": datetime.now().isoformat(),
                }
                papers.append(record)
            except Exception:
                continue

        return papers

    # ─── Step 2: Update Literature DB ─────────────────────────────────────

    def update_literature_db(self, new_papers: List[Dict[str, Any]]) -> int:
        """
        Append new papers to the literature database JSON.

        Returns count of papers added.
        """
        if not new_papers:
            return 0

        # Load existing
        existing = []
        if os.path.exists(self.literature_db_path):
            with open(self.literature_db_path, "r", encoding="utf-8") as f:
                existing = json.load(f)

        # Deduplicate
        added = 0
        for paper in new_papers:
            pmid = str(paper.get("pmid", ""))
            doi = (paper.get("doi", "") or "").lower()
            if pmid and pmid in self.existing_pmids:
                continue
            if doi and doi in self.existing_dois:
                continue
            existing.append(paper)
            if pmid:
                self.existing_pmids.add(pmid)
            if doi:
                self.existing_dois.add(doi)
            added += 1

        # Save
        with open(self.literature_db_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        print(f"[IncrementalUpdater] Added {added} papers to DB (total: {len(existing)})")
        return added

    # ─── Step 3: Incremental FAISS Update ─────────────────────────────────

    def update_faiss_index(
        self,
        new_papers: List[Dict[str, Any]],
        embedding_model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO",
    ) -> int:
        """
        Add new paper embeddings to existing FAISS index incrementally.

        Returns count of vectors added.
        """
        if not new_papers or not faiss:
            return 0

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            try:
                from langchain.embeddings import HuggingFaceEmbeddings
            except ImportError:
                print("[IncrementalUpdater] HuggingFaceEmbeddings not available")
                return 0

        # Load existing index
        index_path = os.path.join(self.faiss_db_dir, "index.faiss")
        if not os.path.exists(index_path):
            print("[IncrementalUpdater] No existing FAISS index found")
            return 0

        try:
            from langchain_community.vectorstores import FAISS
        except ImportError:
            from langchain.vectorstores import FAISS

        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs={"device": "cpu"},
            )
            vectorstore = FAISS.load_local(
                self.faiss_db_dir, embeddings, allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"[IncrementalUpdater] Error loading FAISS: {e}")
            return 0

        # Prepare new documents
        from langchain.schema import Document

        new_docs = []
        for paper in new_papers:
            text = paper.get("abstract", "") or ""
            title = paper.get("title", "") or ""
            if not text and not title:
                continue
            content = f"Title: {title}\nAbstract: {text}" if text else f"Title: {title}"
            metadata = {
                "pmid": paper.get("pmid", ""),
                "doi": paper.get("doi", ""),
                "title": title,
                "year": paper.get("pub_year", ""),
                "journal": paper.get("journal", ""),
                "source": "incremental_update",
            }
            new_docs.append(Document(page_content=content, metadata=metadata))

        if not new_docs:
            return 0

        # Add to FAISS
        vectorstore.add_documents(new_docs)
        vectorstore.save_local(self.faiss_db_dir)
        print(f"[IncrementalUpdater] Added {len(new_docs)} vectors to FAISS index")
        return len(new_docs)

    # ─── Step 4: Tag New Documents (U-Retrieval) ──────────────────────────

    def tag_new_documents(
        self,
        new_papers: List[Dict[str, Any]],
        u_retrieval=None,
    ) -> int:
        """Tag new papers using U-Retrieval tag taxonomy."""
        if not u_retrieval:
            return 0
        count = u_retrieval.tag_documents_batch(new_papers)
        u_retrieval.save_index()
        print(f"[IncrementalUpdater] Tagged {count} new documents")
        return count

    # ─── Step 5: Log Update ───────────────────────────────────────────────

    def log_update(self, stats: Dict[str, Any]):
        """Append update record to log file."""
        log = []
        if os.path.exists(self.update_log_path):
            try:
                with open(self.update_log_path, "r", encoding="utf-8") as f:
                    log = json.load(f)
            except Exception:
                log = []

        record = {
            "timestamp": datetime.now().isoformat(),
            **stats,
        }
        log.append(record)

        os.makedirs(os.path.dirname(self.update_log_path), exist_ok=True)
        with open(self.update_log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)

    # ─── Full Update Pipeline ─────────────────────────────────────────────

    def run_incremental_update(
        self,
        days_back: int = 7,
        update_faiss: bool = True,
        u_retrieval=None,
    ) -> Dict[str, Any]:
        """
        Run the full incremental update pipeline.

        Parameters
        ----------
        days_back : int
            How many days back to search for new papers.
        update_faiss : bool
            Whether to update the FAISS index.
        u_retrieval : URetrieval, optional
            For tagging new documents.

        Returns
        -------
        dict with update statistics.
        """
        stats = {
            "days_back": days_back,
            "started_at": datetime.now().isoformat(),
        }

        # 1. Fetch new papers
        print(f"\n{'='*60}")
        print(f"[IncrementalUpdater] Starting incremental update (last {days_back} days)")
        print(f"{'='*60}")
        new_papers = self.fetch_new_from_pubmed(days_back=days_back)
        stats["papers_fetched"] = len(new_papers)

        if not new_papers:
            stats["status"] = "no_new_papers"
            self.log_update(stats)
            print("[IncrementalUpdater] No new papers found. Done.")
            return stats

        # 2. Update literature DB
        added = self.update_literature_db(new_papers)
        stats["papers_added_to_db"] = added

        # 3. Update FAISS
        if update_faiss:
            vectors_added = self.update_faiss_index(new_papers)
            stats["vectors_added"] = vectors_added

        # 4. Tag documents
        if u_retrieval:
            tagged = self.tag_new_documents(new_papers, u_retrieval)
            stats["docs_tagged"] = tagged

        stats["status"] = "success"
        stats["completed_at"] = datetime.now().isoformat()
        self.log_update(stats)

        print(f"\n{'='*60}")
        print(f"[IncrementalUpdater] Update complete!")
        print(f"  New papers: {len(new_papers)}")
        print(f"  Added to DB: {added}")
        print(f"  Vectors added: {stats.get('vectors_added', 'N/A')}")
        print(f"{'='*60}")

        return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Incremental Literature DB Updater")
    parser.add_argument("--days", type=int, default=7, help="Days to look back")
    parser.add_argument("--no-faiss", action="store_true", help="Skip FAISS update")
    args = parser.parse_args()

    updater = IncrementalUpdater()
    result = updater.run_incremental_update(
        days_back=args.days,
        update_faiss=not args.no_faiss,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
