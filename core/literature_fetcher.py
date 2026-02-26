import os
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

class PubMedFetcher:
    def __init__(self, data_dir="data/pubmed"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def search_articles(self, query, max_results=10, days_back=30):
        """
        Search PubMed for articles matching the query within the last N days.
        """
        print(f"Searching PubMed for: '{query}' (Last {days_back} days, Max: {max_results})")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        date_filter = f' AND ("{start_date.strftime("%Y/%m/%d")}"[Date - Publication] : "{end_date.strftime("%Y/%m/%d")}"[Date - Publication])'
        full_query = query + date_filter
        
        encoded_query = urllib.parse.quote(full_query)
        search_url = f"{self.base_url}esearch.fcgi?db=pubmed&term={encoded_query}&retmode=json&retmax={max_results}"
        
        try:
            response = urllib.request.urlopen(search_url)
            data = json.loads(response.read())
            id_list = data.get("esearchresult", {}).get("idlist", [])
            print(f"Found {len(id_list)} articles.")
            return id_list
        except Exception as e:
            print(f"Error searching PubMed: {e}")
            return []

    def fetch_article_details(self, pmid_list):
        """
        Fetch abstract and metadata for a list of PubMed IDs.
        """
        if not pmid_list:
            return []

        ids_str = ",".join(pmid_list)
        fetch_url = f"{self.base_url}efetch.fcgi?db=pubmed&id={ids_str}&retmode=xml"
        
        articles = []
        try:
            print(f"Requesting details from PubMed for {len(pmid_list)} IDs...")
            response = urllib.request.urlopen(fetch_url, timeout=30)
            xml_data = response.read()
            print("Received XML response. Parsing...")
            root = ET.fromstring(xml_data)
            
            for article in root.findall(".//PubmedArticle"):
                pmid = article.find(".//PMID").text
                
                title_elem = article.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None else "No Title"
                
                # Fetch all AbstractText elements (PMID can have multiple parts)
                abstract_texts = article.findall(".//AbstractText")
                if abstract_texts:
                    abstract_parts = []
                    for elem in abstract_texts:
                        # Use itertext() to get text including mixed content (like <sup>Tags</sup>)
                        text = "".join(elem.itertext())
                        
                        # Handle structured abstracts with Labels (e.g. BACKGROUND: ...)
                        if elem.get("Label"):
                            abstract_parts.append(f"{elem.get('Label')}: {text}")
                        else:
                            abstract_parts.append(text)
                    abstract = " ".join(abstract_parts)
                else:
                    abstract = "No Abstract Available"
                
                # Extract publication year
                pub_date_elem = article.find(".//PubDate/Year")
                year = pub_date_elem.text if pub_date_elem is not None else "Unknown Year"

                # Extract Authors
                authors = []
                for author in article.findall(".//Author"):
                    last_name = author.find("LastName")
                    fore_name = author.find("ForeName")
                    initials = author.find("Initials")
                    
                    name_parts = []
                    if fore_name is not None and fore_name.text:
                        name_parts.append(fore_name.text)
                    if last_name is not None and last_name.text:
                        name_parts.append(last_name.text)
                    elif initials is not None and initials.text: # Fallback if no last name?? usually LastName is there.
                        pass
                        
                    if name_parts:
                        authors.append(" ".join(name_parts))
                
                # Extract Journal
                journal_elem = article.find(".//Journal/Title")
                journal = journal_elem.text if journal_elem is not None else "Unknown Journal"

                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "authors": authors,
                    "journal": journal,
                    "source": "PubMed"
                })
        except Exception as e:
            print(f"Error fetching article details: {e}")
            
        return articles

    def save_articles(self, articles):
        """
        Save fetched articles as text files in the data directory.
        """
        saved_count = 0
        for article in articles:
            filename = f"PMID_{article['pmid']}.txt"
            filepath = os.path.join(self.data_dir, filename)
            
            # Skip if already exists
            if os.path.exists(filepath):
                continue
                
            content = f"Title: {article['title']}\n"
            content += f"PMID: {article['pmid']}\n"
            content += f"Year: {article['year']}\n"
            content += f"Source: {article['source']}\n\n"
            content += f"Abstract:\n{article['abstract']}\n"
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                saved_count += 1
            except Exception as e:
                print(f"Error saving {filename}: {e}")
                
        print(f"Successfully saved {saved_count} new articles to {self.data_dir}")
        return saved_count

    def update_knowledge_base(self, query="CRISPR OR gene editing OR prime editing OR base editing", max_results=50, days_back=30):
        """
        Main pipeline to search, fetch, and save new articles.
        """
        print("--- Starting PubMed Knowledge Base Update ---")
        pmids = self.search_articles(query, max_results, days_back)
        if pmids:
            articles = self.fetch_article_details(pmids)
            self.save_articles(articles)
        print("--- Update Complete ---")

if __name__ == "__main__":
    fetcher = PubMedFetcher()
    # Test with a small fetch
    fetcher.update_knowledge_base(max_results=5, days_back=7)
