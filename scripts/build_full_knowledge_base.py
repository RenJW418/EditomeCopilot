import sys
import os
import json
import time
import hashlib
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.multi_source_fetcher import MultiSourceFetcher

class KnowledgeBaseBuilder:
    def __init__(self, output_dir="data/knowledge_base"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.fetcher = MultiSourceFetcher()
        self.database = {}  # In-memory DB 
        
        # Optimization: Fast lookup sets for O(1) deduplication
        self.seen_dois = set()
        self.seen_titles = set()
        
        # Quarter calculation for versioning
        q = (datetime.now().month - 1) // 3 + 1
        self.version = datetime.now().strftime(f"GEA_v%Y_Q{q}")
        self.previous_version_file = self._find_latest_version()

        # Golden Set for Validation (DOI based)
        self.gold_standard_dois = [
            # Note: These are example DOIs. In a real scenario, you'd add more.
            "10.1126/science.1225829", # Jinek et al. 2012 (CRISPR-Cas9)
            "10.1126/science.1231143", # Cong et al. 2013 (Mammalian cells)
            "10.1038/nature17946",     # Komor et al. 2016 (Base Editing)
            "10.1038/s41586-019-1711-4" # Anzalone et al. 2019 (Prime Editing)
        ]
        
    def _find_latest_version(self):
        """Find the most recent database file to load for incremental updates."""
        try:
            files = [f for f in os.listdir(self.output_dir) if f.startswith("literature_db_GEA_v") and f.endswith(".json")]
            if not files:
                return None
            # Sort by name (which contains date) and pick last
            files.sort()
            return os.path.join(self.output_dir, files[-1])
        except FileNotFoundError:
            return None

    def load_existing_db(self):
        """Load validation for incremental update."""
        if self.previous_version_file and os.path.exists(self.previous_version_file):
            print(f"Loading existing database from {self.previous_version_file}...")
            try:
                with open(self.previous_version_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for article in data:
                        if 'gea_id' in article:
                            self.database[article['gea_id']] = article
                            
                            # Update fast lookup sets
                            if article.get("doi"):
                                self.seen_dois.add(article["doi"].lower().strip())
                            
                            if article.get("title"):
                                norm_title = "".join(x for x in article["title"].lower() if x.isalnum())
                                self.seen_titles.add(norm_title)
                                
                print(f"Loaded {len(self.database)} existing records.")
            except Exception as e:
                print(f"Error loading DB: {e}")

    def generate_queries(self):
        """
        Construct PROFESSIONAL TIERED queries for a high-quality Knowledge Base.
        """
        queries = []

        # --- Tier 1: Core High-Impact Technologies (Base Layer) ---
        # Focus on broad coverage of foundational tools
        # Strategy: "Technology" AND "Gene Editing"
        core_tech = '("CRISPR" OR "Cas9" OR "Cas12" OR "Base Editing" OR "Prime Editing" OR "Gene Editing")'
        # EuropePMC specific: Sort by citation count to get the "Classics" first
        queries.append({
            "q": f'{core_tech} SORT_CITED:Y', 
            "limit": 15000, 
            "tag": "Tier1_Core",
            "source": ["EuropePMC"] # EuropePMC is better for sorting by citation
        })
        
        # --- Tier 2: Recent Frontier (Last 3 Years) ---
        # Focus on latest advancements, even with low citations
        current_year = datetime.now().year
        # Use a more collaborative search range format if possible or adjust dynamically
        # For simplicity, we use EuropePMC syntax here and will handle PubMed specific translation in the loop
        date_range = f"[{current_year-3} TO {current_year}]" 
        frontier_keywords = '("Cas13" OR "CasPhi" OR "Cas7-11" OR "Retron" OR "Transposon-associated" OR "Epigenome Editing" OR "Compact Cas")'
        queries.append({
            "q": f'{frontier_keywords} AND PUB_YEAR:{date_range}', 
            "limit": 8000, 
            "tag": "Tier2_Frontier",
            "source": ["EuropePMC"] # Use EuropePMC for broad date range searches as its syntax is cleaner
        })
        
        # Add a specific PubMed query for recent papers using proper syntax
        queries.append({
            "q": f'{frontier_keywords} AND ({current_year-3}:{current_year}[dp])',
            "limit": 5000,
            "tag": "Tier2_Frontier_PubMed",
            "source": ["PubMed"]
        })

        # --- Tier 3: Delivery & Therapeutics (The "Landing" Problem) ---
        # Critical for "Professional/Applied" KB
        delivery = '("Lipid Nanoparticle" OR "LNP" OR "AAV" OR "Viral Vector" OR "RNP" OR "Electroporation" OR "Exosome")'
        queries.append({
            "q": f'{core_tech} AND {delivery}', 
            "limit": 10000, 
            "tag": "Tier3_Delivery",
            "source": ["PubMed", "EuropePMC"]
        })

        # --- Tier 4: Clinical Translation ---
        # Direct clinical trials and human studies
        queries.append({
            "q": f'{core_tech} AND ("Clinical Trial" OR "Therapy" OR "Patient" OR "In Vivo")', 
            "limit": 5000, 
            "tag": "Tier4_Clinical",
            "source": ["ClinicalTrials", "PubMed"]
        })

        # --- Tier 5: Safety & Off-target ---
        queries.append({
            "q": f'{core_tech} AND ("Off-target" OR "Specificity" OR "Safety" OR "Immunogenicity" OR "Toxicity")', 
            "limit": 5000, 
            "tag": "Tier5_Safety",
            "source": ["PubMed"]
        })

        return queries

    def deduplicate(self, article):
        """
        Deduplication Logic:
        1. DOI (if available) -> Normalize to lowercase
        2. Title (normalized: lowercase, remove non-alphanumeric)
        """
        # Normalize fields
        doi = article.get("doi", "")
        norm_doi = doi.lower().strip() if doi else None
        
        title = article.get("title", "")
        norm_title = "".join(x for x in title.lower() if x.isalnum()) if title else None

        # O(1) Check using sets
        if norm_doi and norm_doi in self.seen_dois:
            return True
        
        if norm_title and norm_title in self.seen_titles:
            return True
        
        # If not found, add to sets immediately IF we decide to add it?
        # No, deduplicate should just check. The caller decides to add to DB.
        
        return False

    def add_article(self, article):
        # Generate ID first to key by ID
        unique_id = None
        if article.get("source") == "PubMed" and article.get("id"):
             unique_id = f"PMID:{article['id']}"
        elif article.get("doi"):
             unique_id = f"DOI:{article['doi']}"
        elif article.get("id") and article.get("source") == "EuropePMC":
             unique_id = f"EPMC:{article['id']}"
        else:
             # Hash title
             unique_id = "HASH:" + hashlib.md5(article.get("title", "").encode()).hexdigest()[:10]
        
        article["gea_id"] = unique_id
        
        # Check if we should update existing record (e.g. to fix missing fields)
        # For now, we always update if ID matches, or if DOI matches
        
        # Deduplication Check
        is_new = True
        
        # Normalize fields
        doi = article.get("doi", "")
        norm_doi = doi.lower().strip() if doi else None
        
        title = article.get("title", "")
        norm_title = "".join(x for x in title.lower() if x.isalnum()) if title else None

        # Check by ID
        if unique_id in self.database:
            # It's an update
            self.database[unique_id].update(article)
            is_new = False
        else:
            # Check by DOI/Title sets
            if (norm_doi and norm_doi in self.seen_dois) or (norm_title and norm_title in self.seen_titles):
                # It's a duplicate found by other means.
                # Find the existing record? Hard with sets.
                # So we skip adding as new.
                # But wait, maybe the new one has better data?
                # Without an index from DOI -> ID, we can't easily update the old record.
                # So we skip.
                return False
            else:
                # Totally new
                article["fetched_at"] = datetime.now().isoformat()
                self.database[unique_id] = article
                is_new = True
        
        # Update lookup sets always
        if norm_doi: self.seen_dois.add(norm_doi)
        if norm_title: self.seen_titles.add(norm_title)
            
        return is_new

    def validate_coverage(self):
        """
        Assess database coverage against Gold Standard and Key Authors.
        """
        print("\n--- Running Coverage Validation ---")
        
        # 1. Gold Standard Check
        found_gold = 0
        for doi in self.gold_standard_dois:
            found = False
            for art in self.database.values():
                if art.get("doi") == doi:
                    found = True
                    break
            if found:
                found_gold += 1
        
        gold_coverage = (found_gold / len(self.gold_standard_dois)) * 100
        print(f"  > Gold Standard Coverage: {gold_coverage:.1f}% ({found_gold}/{len(self.gold_standard_dois)})")
        
        if gold_coverage < 80:
            print("  ! WARNING: Critical landmark papers missing. Consider manual ingestion.")

        return gold_coverage

    def _ingest_gold_standard(self):
        """Force ingest the gold standard papers."""
        print(f"\n--- Ingesting Gold Standard Papers ({len(self.gold_standard_dois)}) ---")
        for doi in self.gold_standard_dois:
            # Check if already exists
            exists = False
            for art in self.database.values():
                if art.get("doi") == doi:
                    exists = True
                    break
            if exists:
                continue

            # Fetch by DOI
            # Europe PMC has good DOI support
            try:
                results = self.fetcher.fetch_europe_pmc(f'DOI:"{doi}"', max_results=1)
                if results:
                    self.add_article(results[0])
                    print(f"  + Added Classic: {doi}")
                else:
                    print(f"  ! Failed to find Classic: {doi}")
            except Exception as e:
                print(f"  ! Error fetching Classic {doi}: {e}")

    def run(self):
        print(f"Starting Knowledge Base Build - Version {self.version}")
        
        # Load previous data for incremental update
        self.load_existing_db()
        
        # Ensure Gold Standard is present
        self._ingest_gold_standard()

        initial_count = len(self.database)

        tiered_queries = self.generate_queries()
        
        # Source Mapping for dynamic calling
        source_map = {
            "PubMed": self.fetcher.fetch_pubmed,
            "EuropePMC": self.fetcher.fetch_europe_pmc,
            "bioRxiv": lambda q, limit: self.fetcher.fetch_europe_pmc(f'{q} AND SRC:PPR', limit),
            "ClinicalTrials": self.fetcher.fetch_clinical_trials
        }

        total_fetched = 0
        newly_added = 0

        for t_idx, tier in enumerate(tiered_queries):
            query = tier["q"] if isinstance(tier, dict) else tier
            limit = tier.get("limit", 2000)
            sources = tier.get("source", ["PubMed"]) # Default to PubMed if not specified
            tag = tier.get("tag", f"Tier_{t_idx}")
            
            print(f"\n[[ Processing {tag} ]]")
            print(f"Query: {query[:80]}... (Limit: {limit})")
            
            for src_name in sources:
                fetch_func = source_map.get(src_name)
                if not fetch_func:
                    print(f"  ! Unknown source: {src_name}")
                    continue
                    
                print(f"  > Fetching from {src_name}...")
                try:
                    # Clean query for specific sources if needed (e.g. remove sort commands for PubMed?)
                    # PubMed doesn't support 'SORT_CITED:Y' in query string broadly via ESearch term, 
                    # but EuropePMC does. PubMed ignores unrecognized terms usually or returns 0.
                    # We should be careful.
                    
                    time.sleep(1) # Graceful delay
                    
                    results = fetch_func(query, max_results=limit)
                    print(f"  > {src_name} returned {len(results)} results")
                    
                    batch_added = 0
                    for i, res in enumerate(results):
                        # Tag article with tier info
                        res["tier_tag"] = tag
                        if self.add_article(res):
                            newly_added += 1
                            batch_added += 1
                        
                        # Periodically save during processing of large batches
                        if newly_added > 0 and newly_added % 500 == 0:
                            print(f"    (Auto-saving progress at {newly_added} records...)")
                            self.save_database()

                    total_fetched += len(results)
                    print(f"  > Added {batch_added} new articles.")
                    
                    # Save after each source is fully processed
                    self.save_database()
                        
                except Exception as e:
                    print(f"  ! Error in {src_name}: {e}")
                    
        self.save_database()
        self.validate_coverage()
        self.generate_report(initial_count, total_fetched, newly_added)

    def save_database(self):
        filename = f"literature_db_{self.version}.json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(list(self.database.values()), f, indent=2, ensure_ascii=False)
        print(f"\nSaved database to {filepath}")

    def generate_report(self, initial, fetched, added):
        report_content = []
        report_content.append(f"# Gene Editing Knowledge Base Build Report")
        report_content.append(f"**Version**: {self.version}")
        report_content.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append(f"")
        report_content.append(f"## 1. Statistics")
        report_content.append(f"- Initial DB Size: {initial}")
        report_content.append(f"- Total Fetched in this run: {fetched}")
        report_content.append(f"- Newly Added: {added}")
        report_content.append(f"- Total Database Size: {len(self.database)}")
        report_content.append(f"")
        report_content.append(f"## 2. Methodology")
        report_content.append(f"### Search Strategy")
        report_content.append(f"- **Layered Queries**: 4 distinct boolean query layers covering Technology, Mechanism, Delivery, and RNA Editing.")
        report_content.append(f"- **Sources**: PubMed, Europe PMC, bioRxiv (via PMC), ClinicalTrials.gov")
        report_content.append(f"### Deduplication")
        report_content.append(f"- Primary Key: DOI (Normalized)")
        report_content.append(f"- Secondary Key: Title Hash")
        report_content.append(f"")
        report_content.append(f"## 3. Coverage Validation")
        
        # Calculate coverage again for report
        found_gold = 0
        for doi in self.gold_standard_dois:
            for art in self.database.values():
                if art.get("doi") == doi:
                    found_gold += 1
                    break
        gold_coverage = (found_gold / len(self.gold_standard_dois)) * 100
        
        report_content.append(f"- **Gold Standard Coverage**: {gold_coverage:.1f}% ({found_gold}/{len(self.gold_standard_dois)})")
        if gold_coverage < 100:
            report_content.append(f"- **Risk Analysis**: Some landmark papers are missing. Recommendation: Run targeted ingestion for missing DOIs.")
        else:
            report_content.append(f"- **Assessment**: High confidence in core coverage.")

        report_path = os.path.join(self.output_dir, f"report_{self.version}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_content))
        
        print(f"\nReport saved to {report_path}")
        print("\n==================================")
        print(f"BUILD REPORT: {self.version}")
        print("==================================")
        print(f"Total Database Size: {len(self.database)}")
        print("==================================")

if __name__ == "__main__":
    builder = KnowledgeBaseBuilder()
    builder.run()
