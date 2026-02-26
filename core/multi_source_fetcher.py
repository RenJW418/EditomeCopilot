import requests
import time
import urllib.parse
from datetime import datetime
import xml.etree.ElementTree as ET

class MultiSourceFetcher:
    def __init__(self):
        self.headers = {
            "User-Agent": "GeneEditingAlmanac/1.0 (mailto:admin@example.com)"
        }

    def _make_request(self, method, url, **kwargs):
        """Helper to make requests with retry logic."""
        max_retries = 3
        backoff = 1
        last_exception = None

        for i in range(max_retries):
            try:
                if method.lower() == 'get':
                    resp = requests.get(url, **kwargs)
                else:
                    resp = requests.post(url, **kwargs)
                
                # Check for 429 Too Many Requests specifically
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", backoff * 4)) # EPMC likes high wait
                    print(f"    ! Rate limited (429). Retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                
                if resp.status_code >= 500: # Server error
                    print(f"    ! Server error {resp.status_code}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                resp.raise_for_status()
                return resp

            except (requests.exceptions.RequestException, requests.exceptions.ChunkedEncodingError) as e:
                last_exception = e
                if i == max_retries - 1:
                    print(f"    ! Request failed finally after {max_retries} tries: {e}")
                    raise e
                
                print(f"    ! Request failed ({e}). Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
        
        return None

    def fetch_pubmed(self, query, max_results=20):
        print(f"Fetching from PubMed: {query} (Target: {max_results})")
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
        # 1. ESearch
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "usehistory": "y" 
        }
        
        try:
            search_url = f"{base_url}esearch.fcgi"
            response = self._make_request('get', search_url, params=search_params, headers=self.headers, timeout=30)
            data = response.json()
            
            esearch_result = data.get("esearchresult", {})
            id_list = esearch_result.get("idlist", [])
            webenv = esearch_result.get("webenv")
            query_key = esearch_result.get("querykey")
            count = int(esearch_result.get("count", 0))
            
            print(f"  > PubMed found {count} total matches.")
            
            if not id_list and count == 0:
                return []
            
            # EFetch in Batches using History
            # Use retstart max 9999? No, with WebEnv it should support more.
            # But just in case, let's limit fetching to 9999 PER QUERY for stability if > 10k fails.
            # OR we try to fetch by ID list if < 10k? No, that's what we have.
            # If > 10k fails, we can't do much without complex logic (splitting query by date).
            
            # Initialize effective_max based on available results and requested max
            effective_max = min(count, max_results)
            
            # SAFEGUARD: If effective_max > 9998, cap it to 9998 for PubMed
            # Because deep paging in PubMed via HTTP POST often unstable or restricted.
            if effective_max > 9998:
                print("  > Cap PubMed at 9998 for API stability.")
                effective_max = 9998
            
            # Initialize results list
            results = []
            
            current_batch_size = 200 # Reset inside
            
            for start in range(0, effective_max, current_batch_size):
                # Adjust batch size for last batch
                real_batch_size = min(current_batch_size, effective_max - start)
                print(f"    (Processing batch {start}-{start+real_batch_size} of {effective_max}...)")
                
                fetch_url = f"{base_url}efetch.fcgi"
                fetch_params = {
                    "db": "pubmed",
                    "retmode": "xml",
                    "WebEnv": webenv,
                    "query_key": query_key,
                    "retstart": start,
                    "retmax": real_batch_size
                }

                try:
                    resp = self._make_request('post', fetch_url, data=fetch_params, headers=self.headers, timeout=60)
                    if not resp:
                        # Should have raised in _make_request if retries exhausted
                        continue
                        
                    root = ET.fromstring(resp.content)
                    articles = root.findall(".//PubmedArticle")
                    if not articles:
                        articles = []

                    for article in articles:
                        pmid = article.findtext(".//PMID")
                        title = article.findtext(".//ArticleTitle")
                        
                        # Enhanced Abstract Extraction
                        abstract_texts = article.findall(".//AbstractText")
                        if abstract_texts:
                            abstract_parts = []
                            for elem in abstract_texts:
                                # Start with element text
                                text = "".join(elem.itertext())
                                if elem.get("Label"):
                                    abstract_parts.append(f"{elem.get('Label')}: {text}")
                                else:
                                    abstract_parts.append(text)
                            abstract = " ".join(abstract_parts)
                        elif article.find(".//Abstract"):
                            # Fallback for simple abstract
                            abstract = "".join(article.find(".//Abstract").itertext())
                        else:
                            abstract = ""
                            
                        doi = article.findtext(".//ArticleId[@IdType='doi']")
                        
                        # Extract Year
                        pub_date = article.find(".//PubDate")
                        year = ""
                        if pub_date is not None:
                            year = pub_date.findtext("Year")
                            if not year:
                                # Try MedlineDate
                                medline = pub_date.findtext("MedlineDate")
                                if medline:
                                    year = medline.split()[0] # "2020 Oct-Dec" -> "2020"
                        
                        # Extract Journal
                        journal = article.findtext(".//Journal/Title")
                        
                        # Extract Authors
                        authors = []
                        author_list = article.find(".//AuthorList")
                        if author_list is not None:
                            for author in author_list.findall("Author"):
                                last = author.findtext("LastName")
                                fore = author.findtext("ForeName")
                                initials = author.findtext("Initials")
                                
                                name = ""
                                if last and fore:
                                    name = f"{fore} {last}"
                                elif last and initials:
                                    name = f"{initials} {last}"
                                elif last:
                                    name = last
                                
                                # Collective Name (Consortium)
                                if not name:
                                    collective = author.findtext("CollectiveName")
                                    if collective:
                                        name = collective
                                        
                                if name:
                                    # Standardize author format (Last F)
                                    parts = name.split()
                                    if len(parts) > 1:
                                        standardized_name = f"{parts[-1]} {parts[0][0]}"
                                    else:
                                        standardized_name = name
                                    authors.append(standardized_name)

                        # Skip if essential metadata missing (e.g. no title or abstract)
                        if not title or len(title) < 5:
                            continue

                        results.append({
                            "source": "PubMed",
                            "id": pmid,
                            "doi": doi,
                            "title": title,
                            "abstract": abstract,
                            "year": year,
                            "journal": journal,
                            "authors": authors,
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                            "gea_id": f"PMID:{pmid}",
                            "fetched_at": datetime.now().isoformat()
                        })
                    
                    print(f"    > Fetched batch {start}-{start+len(articles)}...")
                    time.sleep(0.5) 
                    
                except Exception as e:
                    print(f"    ! Error fetching batch starting at {start}: {e}")
                    
            return results

        except Exception as e:
            print(f"PubMed Error: {e}")
            return []

    def fetch_europe_pmc(self, query, max_results=20):
        print(f"Fetching from Europe PMC: {query} (Target: {max_results})")
        base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        results = []
        cursor_mark = "*"
        batch_size = 1000 
        
        fetched_count = 0
        
        while fetched_count < max_results:
            remaining = max_results - fetched_count
            current_page_size = min(batch_size, remaining)
            # API max page size is 1000
            if current_page_size > 1000: current_page_size = 1000
            
            params = {
                "query": query,
                "format": "json",
                "pageSize": current_page_size,
                "cursorMark": cursor_mark,
                "resultType": "core"
            }
            
            try:
                triggered_break = False
                response = self._make_request('get', base_url, params=params, headers=self.headers, timeout=30)
                if not response:
                    break
                    
                data = response.json()
                items = data.get("resultList", {}).get("result", [])
                
                if not items:
                    triggered_break = True
                
                if not triggered_break:
                    for item in items:
                        # Extract Authors
                        authors = []
                        if "authorString" in item:
                            # "Smith J, Doe A"
                            authors = [a.strip() for a in item["authorString"].split(",")]
                        elif "authorList" in item:
                            for auth in item["authorList"].get("author", []):
                                if "fullName" in auth:
                                    authors.append(auth["fullName"])
                                elif "lastName" in auth:
                                    authors.append(f"{auth.get('firstName', '')} {auth['lastName']}")

                        # Extract Journal
                        journal_title = item.get("journalTitle", "")
                        if not journal_title:
                            journal_info = item.get("journalInfo", {})
                            journal = journal_info.get("journal", {})
                            journal_title = journal.get("title", "") or journal.get("medlineAbbreviation", "")

                        results.append({
                            "source": "EuropePMC",
                            "id": item.get("id"),
                            "doi": item.get("doi"),
                            "title": item.get("title"),
                            "abstract": item.get("abstractText", ""),
                            "year": item.get("pubYear", ""),
                            "journal": journal_title,
                            "authors": authors,
                            "url": f"https://europepmc.org/article/{item.get('source')}/{item.get('id')}",
                            "gea_id": item.get("id") or item.get("doi") or f"EPMC:{item.get('id')}",
                            "fetched_at": datetime.now().isoformat()
                        })
                

                if triggered_break:
                     break
                
                fetched_count += len(items)
                next_cursor = data.get("nextCursorMark")
                
                print(f"  > Fetched {fetched_count}/{max_results} from EuropePMC...")
                
                if fetched_count >= max_results:
                     break

                if cursor_mark == next_cursor:
                    break # No more pages
                cursor_mark = next_cursor
                
                # Dynamic sleep 
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Europe PMC Batch Error: {e}")
                break
                
        return results

    def fetch_biorxiv(self, query, max_results=20):
        # Using Europe PMC for bioRxiv content
        print(f"Fetching bioRxiv via Europe PMC (source:PPR)...")
        return self.fetch_europe_pmc(f'{query} AND SRC:PPR', max_results)

    def fetch_clinical_trials(self, query, max_results=20):
        print(f"Fetching from ClinicalTrials.gov: {query}")
        # Simplified implementation for batch retrieval
        base_url = "https://clinicaltrials.gov/api/v2/studies"
        results = []
        next_page_token = None
        fetched_count = 0
        batch_size = 20 # Smaller batch size for stability
        
        while fetched_count < max_results:
            params = {
                "query.term": query,
                "pageSize": str(min(batch_size, max_results - fetched_count))
            }
            if next_page_token:
                params["pageToken"] = next_page_token
            
            try:
                response = self._make_request('get', base_url, params=params, headers=self.headers, timeout=30)
                if not response:
                    break
                    
                data = response.json()
                studies = data.get("studies", [])
                
                if not studies:
                    break
                    
                for study in studies:
                    protocol = study.get("protocolSection", {})
                    id_module = protocol.get("identificationModule", {})
                    status_module = protocol.get("statusModule", {})
                    
                    nct_id = id_module.get("nctId")
                    title = id_module.get("briefTitle")
                    official_title = id_module.get("officialTitle")
                    status = status_module.get("overallStatus")
                    
                    brief_summary = protocol.get("descriptionModule", {}).get("briefSummary", "")
                    
                    # Better abstract construction
                    abstract = f"Status: {status}\n\nSummary: {brief_summary}"
                    
                    start_date_struct = status_module.get("startDateStruct", {})
                    year = start_date_struct.get("date", "").split("-")[0] if start_date_struct.get("date") else ""

                    sponsor = protocol.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", "Unknown Sponsor")

                    results.append({
                        "source": "ClinicalTrials.gov",
                        "id": nct_id,
                        "doi": None, # CT.gov doesn't usually have DOI
                        "title": title or official_title,
                        "abstract": abstract,
                        "year": year,
                        "journal": "ClinicalTrials.gov",
                        "authors": [sponsor], # Use sponsor as author
                        "url": f"https://clinicaltrials.gov/study/{nct_id}",
                        "gea_id": f"NCT:{nct_id}",
                        "fetched_at": datetime.now().isoformat()
                    })
                
                fetched_count += len(studies)
                next_page_token = data.get("nextPageToken")
                
                if not next_page_token:
                    break
                    
                print(f"  > Fetched {fetched_count}/{max_results} from ClinicalTrials...")
                time.sleep(1.0) # Respectful delay

            except Exception as e:
                print(f"ClinicalTrials Search Error: {e}")
                break
                
        return results
