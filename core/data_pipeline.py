import os
import json
import hashlib
import requests
import pdfplumber
import time
import re
from datetime import datetime
import bibtexparser
import rispy
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from core.llm_client import LLMClient
from core.knowledge_graph import GeneEditingKnowledgeGraph

class GeneEditingDataPipeline:
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        self.pdf_dir = os.path.join(base_dir, "pdfs")
        self.db_dir = os.path.join(base_dir, "faiss_db")
        self.user_db_dir = os.path.join(base_dir, "user_uploads_db")
        os.makedirs(self.pdf_dir, exist_ok=True)
        os.makedirs(self.db_dir, exist_ok=True)
        os.makedirs(self.user_db_dir, exist_ok=True)
        
        self.llm = LLMClient()
        self.kg = GeneEditingKnowledgeGraph(persistence_path=os.path.join(base_dir, "knowledge_base/kg.json"))
        
        # Step 5: Embeddings
        # Priority: SPECTER2 (biomedical, citation-aware) → BiomedBERT → MiniLM fallback
        # Override via EMBEDDING_MODEL env var.
        _default_model = os.getenv(
            "EMBEDDING_MODEL",
            "pritamdeka/S-PubMedBert-MS-MARCO",  # Best biomedical retrieval model
        )
        _fallbacks = [
            _default_model,
            "pritamdeka/S-PubMedBert-MS-MARCO",   # Strong biomedical retrieval
            "dmis-lab/biobert-base-cased-v1.2",    # Classic BioNLP model
            "sentence-transformers/all-MiniLM-L6-v2",  # General fallback
        ]
        self.embeddings = None
        for model_name in _fallbacks:
            try:
                print(f"[Embeddings] Trying model: {model_name}")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": os.getenv("EMBED_DEVICE", "cpu")},
                    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
                )
                print(f"[Embeddings] Loaded: {model_name}")
                break
            except Exception as e:
                print(f"[Embeddings] Failed to load {model_name}: {e}")
        if self.embeddings is None:
            raise RuntimeError("No embedding model could be loaded.")
        self.vector_store = None
        self.user_vector_store = None
        self.bm25 = None
        self.bm25_corpus = [] # List of Document objects
        self.user_bm25 = None
        self.user_bm25_corpus = []

        self._load_index()

    def _load_index(self):
        """Step 6: Load Vector DB Index if exists"""
        # Load Main Index
        if os.path.exists(os.path.join(self.db_dir, "index.faiss")):
            self.vector_store = FAISS.load_local(self.db_dir, self.embeddings, allow_dangerous_deserialization=True)
            print(f"Loaded existing Main FAISS index from {self.db_dir}")
            self._load_bm25("main")
        else:
            print("No existing Main FAISS index found.")

        # Load User Uploads Index
        if os.path.exists(os.path.join(self.user_db_dir, "index.faiss")):
            self.user_vector_store = FAISS.load_local(self.user_db_dir, self.embeddings, allow_dangerous_deserialization=True)
            print(f"Loaded existing User FAISS index from {self.user_db_dir}")
            self._load_bm25("user")
        else:
            print("No existing User FAISS index found.")
            
    def _load_bm25(self, target="main"):
        """Load or Rebuild BM25 Index"""
        base = self.db_dir if target == "main" else self.user_db_dir
        corpus_path = os.path.join(base, "bm25_corpus.pkl")
        
        if os.path.exists(corpus_path):
            try:
                with open(corpus_path, 'rb') as f:
                    corpus = pickle.load(f)
                
                tokenized_corpus = [doc.page_content.lower().split() for doc in corpus]
                bm25_index = BM25Okapi(tokenized_corpus)
                
                if target == "main":
                    self.bm25_corpus = corpus
                    self.bm25 = bm25_index
                else:
                    self.user_bm25_corpus = corpus
                    self.user_bm25 = bm25_index
                    
                print(f"Loaded BM25 index for {target} with {len(corpus)} documents.")
            except Exception as e:
                print(f"Error loading BM25 for {target}: {e}")

    def step0_planning(self):
        """Step 0: Planning & Scope"""
        # Define the query strategy (CRISPR, Cas9, prime editing, etc.)
        query = '("CRISPR" OR "Cas9" OR "prime editing" OR "base editing" OR "TALEN" OR "ZFN" OR "Cas13" OR "ADAR") AND OPEN_ACCESS:Y'
        return query

    def step1_fetch_metadata(self, query, max_results=10):
        """Step 1: Fetch Metadata & OA Links using Europe PMC (or Local JSON if available)"""
        # [Optimization] Check for local large JSON dump first
        local_db_path = os.path.join(self.base_dir, "knowledge_base/literature_db_GEA_v2026_Q1.json")
        if os.path.exists(local_db_path):
            print(f"\n[Step 1] Found local literature database: {local_db_path}")
            try:
                with open(local_db_path, "r", encoding="utf-8") as f:
                    # Load efficient subset or generator if file is huge?
                    # For 100k records (~200MB), loading into memory is fine on modern servers.
                    data = json.load(f)
                    print(f"Loaded {len(data)} records from local database.")
                    
                    # Convert to standard metadata format
                    metadata_list = []
                    # Simple keyword filtering to simulate 'query' roughly, or just return top N relevant?
                    # Since we have a local DB, let's treat it as the source of truth and return the requested slice
                    # or filter by query keywords if query is specific.
                    
                    # MVP: If query is generic ("CRISPR..."), return top N from local DB 
                    # (Assuming they are already relevant gene editing papers)
                    count = 0
                    for item in data:
                        if count >= max_results:
                            break
                        
                        # Check if matches query loosely (case insensitive)
                        # Split query into keywords, remove OR/AND syntax roughly
                        keywords = [k.strip('"()') for k in query.split() if len(k) > 4 and k not in ["AND", "OR"]]
                        text_content = (item.get("title", "") + " " + item.get("abstract", "")).lower()
                        
                        # If query is complex, just take the items (assuming DB is already domain specific)
                        # For "top-tier" project, we assume the DB is CURATED for Gene Editing.
                        # So we process them.
                        
                        meta = {
                            "doc_id": item.get("id", "") or item.get("pmid", "") or item.get("doi", ""),
                            "doi": item.get("doi", ""),
                            "pmid": item.get("id", ""), # Assuming ID is PMID
                            "pmcid": item.get("pmcid", ""),
                            "title": item.get("title", ""),
                            "authors": item.get("authors", "Unknown"), # JSON might not have authors?
                            "journal": item.get("journal", ""),
                            "year": item.get("year", ""),
                            "oa_status": "local-json",
                            "pdf_url": None, # Local JSON usually doesn't have PDF links or we don't need them
                            "abstract": item.get("abstract", ""),
                            "source_type": "local_db"
                        }
                        
                        # Only add if it has an abstract (crucial for content)
                        if meta["abstract"]:
                             metadata_list.append(meta)
                             count += 1
                             
                    print(f"Selected {len(metadata_list)} relevant records from local DB.")
                    return metadata_list
            except Exception as e:
                print(f"Error reading local DB: {e}. Falling back to API.")
        
        print(f"\n[Step 1] Fetching metadata from Europe PMC for query: {query}")
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={query}&format=json&resultType=core&pageSize={max_results}"
        
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            data = response.json()
            results = data.get("resultList", {}).get("result", [])
            
            metadata_list = []
            for item in results:
                # Extract OA PDF URL if available
                pdf_url = None
                full_text_urls = item.get("fullTextUrlList", {}).get("fullTextUrl", [])
                for ft_url in full_text_urls:
                    if ft_url.get("documentStyle") == "pdf":
                        pdf_url = ft_url.get("url")
                        break
                
                meta = {
                    "doc_id": item.get("id", ""),
                    "doi": item.get("doi", ""),
                    "pmid": item.get("pmid", ""),
                    "pmcid": item.get("pmcid", ""),
                    "title": item.get("title", ""),
                    "authors": item.get("authorString", ""),
                    "journal": item.get("journalTitle", ""),
                    "year": item.get("pubYear", ""),
                    "oa_status": item.get("isOpenAccess", "N"),
                    "pdf_url": pdf_url,
                    "abstract": item.get("abstractText", "")
                }
                metadata_list.append(meta)
            print(f"Found {len(metadata_list)} articles with metadata.")
            return metadata_list
        except Exception as e:
            print(f"Error fetching metadata: {e}")
            return []

    def step2_download_pdfs(self, metadata_list):
        """Step 2: Fetch PDF (Download OA PDFs) or skip if local DB"""
        print("\n[Step 2] Downloading PDFs...")
        downloaded_files = []
        
        # If data came from local JSON DB, we skip PDF download and pass metadata directly to step 3
        # We mark them so step 3 knows to use 'abstract' as text
        skipped_for_local = []
        
        # More resilient HTTP settings for unstable networks
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        max_retries = 3
        connect_timeout = 10
        read_timeout = 60
        
        for meta in metadata_list:
            # CHECK: If from local DB, skip download
            if meta.get("source_type") == "local_db":
                skipped_for_local.append(meta)
                continue
                
            if not meta["pdf_url"]:
                continue
                
            filename = f"{meta['pmcid'] or meta['pmid'] or meta['doc_id']}.pdf"
            filepath = os.path.join(self.pdf_dir, filename)
            
            if os.path.exists(filepath):
                print(f"Skipping {filename}, already exists.")
                downloaded_files.append((filepath, meta))
                continue

            # Build fallback URL list: Europe PMC URL first, then NCBI PMC URL if PMCID exists.
            candidate_urls = [meta["pdf_url"]]
            pmcid = meta.get("pmcid")
            if pmcid:
                candidate_urls.append(f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf")
                
            try:
                download_ok = False
                for url_idx, pdf_url in enumerate(candidate_urls):
                    for attempt in range(1, max_retries + 1):
                        try:
                            print(f"Downloading {filename} from {pdf_url} (attempt {attempt}/{max_retries})...")
                            response = requests.get(
                                pdf_url,
                                headers=headers,
                                stream=True,
                                timeout=(connect_timeout, read_timeout),
                                allow_redirects=True,
                            )

                            if response.status_code != 200:
                                raise requests.HTTPError(f"HTTP {response.status_code}")

                            with open(filepath, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)

                            downloaded_files.append((filepath, meta))
                            download_ok = True
                            break

                        except Exception as e:
                            print(f"Attempt failed for {filename}: {e}")
                            # Exponential backoff: 1s, 2s, 4s
                            if attempt < max_retries:
                                time.sleep(2 ** (attempt - 1))

                    if download_ok:
                        break
                    # If this URL failed all retries, switch to next fallback URL
                    if url_idx < len(candidate_urls) - 1:
                        print(f"Switching download source for {filename}...")

                if not download_ok:
                    print(f"Failed to download {filename} after retries and fallbacks.")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                
        # If We skipped PDF download for local DB, we need to pass them to step 3 as well
        # The skipped_for_local are appended here
        for meta in skipped_for_local:
             downloaded_files.append(("LOCAL_JSON_SOURCE", meta))

        # Check: If we have NO downloads and NO local files, warn user
        if not downloaded_files:
             print("Warning: No documents available for processing (download failed or no local matches).")
             
        if skipped_for_local:
            print(f"Skipped PDF download for {len(skipped_for_local)} local database records.")

        return downloaded_files

    def step3_parse_pdfs(self, downloaded_files):
        """Step 3: Structured Parsing (PDF -> Text) or JSON Text Extraction"""
        print("\n[Step 3] Parsing Documents (PDFs or Local JSON)...")
        parsed_docs = []
        
        if not downloaded_files:
            print("No files to parse.")
            return []

        for filepath, meta in downloaded_files:
            # CASE 1: Local JSON Record (Abstract Context)
            if filepath == "LOCAL_JSON_SOURCE":
                # Construct a rich text representation from metadata + abstract
                # This ensures the Embedding model gets context (Title + Abstract)
                text = f"Title: {meta.get('title', '')}\n"
                text += f"Authors: {meta.get('authors', 'Unknown')}\n"
                text += f"Journal: {meta.get('journal', 'Unknown')} ({meta.get('year', '')})\n"
                text += f"DOI: {meta.get('doi', '')}\n"
                text += f"PMID: {meta.get('pmid', '')}\n\n"
                text += "Abstract:\n"
                text += meta.get('abstract', '')
                
                if text.strip():
                    meta["parsed_version"] = "local_json_v1"
                    parsed_docs.append({"text": text, "metadata": meta})
                continue

            # CASE 2: PDF File
            try:
                text = ""
                with pdfplumber.open(filepath) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                
                if text.strip():
                    meta["parsed_version"] = "pdfplumber_v0.11"
                    parsed_docs.append({"text": text, "metadata": meta})
                    print(f"Successfully parsed {os.path.basename(filepath)}")
            except Exception as e:
                print(f"Error parsing {filepath}: {e}")
                
        return parsed_docs

    def step3_5_structured_extraction(self, parsed_docs):
        """Step 3.5: LLM-based Structured Extraction (Capability Matrix, Efficiency, Context, Delivery, Evidence Level)"""
        print("\n[Step 3.5] Performing Structured Extraction using LLM...")
        if not self.llm.client:
            print("LLM API not configured. Skipping structured extraction.")
            return parsed_docs

        prompt_template = """
        Extract the following gene editing information from the provided text.
        Return ONLY a valid JSON object with the following keys:
        - "technology": (e.g., CRISPR KO, Base Editing, Prime Editing, RNA editing)
        - "cas_type": (e.g., SpCas9, SaCas9, Cas12a, Cas13)
        - "mutation_type": (e.g., SNV, insertion, deletion)
        - "efficiency": (e.g., 65%, or specific indel rate)
        - "off_target_risk": (e.g., low, high, or specific frequency)
        - "cell_type": (e.g., HEK293, HSC, primary cell)
        - "species": (e.g., human, mouse)
        - "delivery_system": (e.g., AAV, LNP, electroporation)
        - "evidence_level": (Level 1: Clinical, Level 2: Animal, Level 3: In vitro, Level 4: Predictive)

        If a field is not mentioned, use "Unknown".

        Text:
        {text}
        """

        for doc in parsed_docs:
            try:
                # Use the first 3000 characters to avoid token limits for extraction
                text_sample = doc["text"][:3000]
                prompt = prompt_template.format(text=text_sample)
                response = self.llm.generate(prompt, system_prompt="You are a precise data extraction agent for gene editing literature. Output only JSON.")

                extracted_data = self._safe_parse_json(response)
                if not isinstance(extracted_data, dict):
                    raise ValueError("LLM extraction response is not valid JSON")
                
                # Attach extracted data to metadata
                doc["metadata"]["structured_data"] = extracted_data
                
                # Ingest into Knowledge Graph
                if extracted_data:
                    study_id = doc['metadata'].get('pmid') or doc['metadata'].get('doi') or f"Doc_{hash(doc['metadata'].get('title', 'Unknown'))}"
                    self.kg.ingest_structured_data(extracted_data, str(study_id))
                    
                print(f"Extracted structured data for {doc['metadata'].get('pmid', 'doc')}")
            except Exception as e:
                print(f"Error extracting structured data for {doc['metadata'].get('pmid', 'doc')}: {e}")
                doc["metadata"]["structured_data"] = {}
        
        # Save KG after batch processing
        self.kg.save_graph()
        return parsed_docs


    def _safe_parse_json(self, text):
        if not text:
            return None

        cleaned = str(text).replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                if len(match.group(0)) > 2:
                   return json.loads(match.group(0))
            except Exception:
                return None
        return None

    def import_user_library(self, file_content, file_format='bibtex'):
        """Import references from user-provided file (BibTeX or RIS)"""
        parsed_docs = []
        import hashlib
        
        try:
            if file_format == 'bibtex':
                library = bibtexparser.loads(file_content)
                entries = library.entries
                print(f"[Import] Parsed {len(entries)} BibTeX entries.")
                
                for entry in entries:
                    abstract = entry.get('abstract', '') 
                    if not abstract: continue # Skip empty abstracts often found in exports
                    
                    doc_meta = {
                        "doi": entry.get('doi', ''),
                        "pmid": entry.get('pmid', ''),
                        "pmcid": "",
                        "title": entry.get('title', 'UserId').replace('{', '').replace('}', ''),
                        "authors": entry.get('author', '').replace('\n', ' '),
                        "journal": entry.get('journal', ''),
                        "year": entry.get('year', 'n.d.'),
                        "oa_status": "user-import",
                        "parsed_version": "1.0",
                        "doc_id": entry.get('ID', f"user_{hashlib.md5(entry.get('title', '').encode()).hexdigest()[:10]}")
                    }
                    
                    doc = {
                        "text": f"Title: {doc_meta['title']}\nAuthors: {doc_meta['authors']}\n\n{abstract}",
                        "metadata": doc_meta
                    }
                    parsed_docs.append(doc)
                    
            elif file_format == 'ris':
                entries = rispy.loads(file_content)
                print(f"[Import] Parsed {len(entries)} RIS entries.")
                
                for entry in entries:
                    abstract = entry.get('abstract', '') or entry.get('notes_abstract', '')
                    if not abstract: continue 
                    
                    doc_meta = {
                        "doi": entry.get('doi', ''),
                        "pmid": "", 
                        "pmcid": "",
                        "title": entry.get('title', 'Untitled'),
                        "authors": ", ".join(entry.get('authors', [])),
                        "journal": entry.get('journal_name', ''),
                        "year": entry.get('year', 'n.d.'),
                        "oa_status": "user-import",
                        "parsed_version": "1.0",
                        "doc_id": f"user_ris_{hashlib.md5(entry.get('title', '').encode()).hexdigest()[:10]}"
                    }
                     
                    doc = {
                        "text": f"Title: {doc_meta['title']}\nAuthors: {doc_meta['authors']}\n\n{abstract}",
                        "metadata": doc_meta
                    }
                    parsed_docs.append(doc)

            if parsed_docs:
                # Reuse the existing pipeline starting from step 4
                print(f"[Import] Processing {len(parsed_docs)} valid documents...")
                # We need to temporarily adapt parsed_docs to match step4 expectations
                # step4 expects list of dicts with 'text' and 'metadata' keys, which we prepared.
                
                # Check formatting of metadata for chunking
                # The step4_chunking iterates parsed_docs and splits text.
                chunks = self.step4_chunking(parsed_docs)
                self.step5_6_embed_and_index(chunks, target="user")
                return len(chunks)
            else:
                return 0

        except Exception as e:
            print(f"Error importing library: {e}")
            return 0

    def step4_chunking(self, parsed_docs):
        """Step 4: Chunking and Deduplication"""
        print("\n[Step 4] Chunking text (256-512 tokens, 10-20% overlap)...")
        # Section-aware chunking: smaller chunks for precise retrieval
        # chunk_size ~800 chars ≈ 200 tokens; overlap ~160 chars ensures context continuity
        # Separators respect abstract sections (Methods / Results / Conclusion etc.)
        _chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
        _chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "160"))
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=_chunk_size,
            chunk_overlap=_chunk_overlap,
            length_function=len,
            separators=[
                "\n## ", "\n### ",        # Markdown section headers
                "INTRODUCTION:", "METHODS:", "RESULTS:", "CONCLUSION:",  # structured abstract
                "Introduction:", "Methods:", "Results:", "Conclusion:",
                "\n\n", "\n", ". ", " ", "",
            ],
        )
        
        chunks = []
        for doc in parsed_docs:
            splits = text_splitter.split_text(doc["text"])
            for i, split in enumerate(splits):
                # Generate checksum for deduplication
                checksum = hashlib.sha256(split.encode('utf-8')).hexdigest()
                
                # Rich metadata schema
                chunk_meta = {
                    "doc_id": doc["metadata"]["doc_id"],
                    "doi": doc["metadata"]["doi"],
                    "pmid": doc["metadata"]["pmid"],
                    "pmcid": doc["metadata"]["pmcid"],
                    "title": doc["metadata"]["title"],
                    "authors": doc["metadata"].get("authors", ""),
                    "journal": doc["metadata"]["journal"],
                    "year": doc["metadata"]["year"],
                    "oa_status": doc["metadata"]["oa_status"],
                    "parsed_version": doc["metadata"]["parsed_version"],
                    "chunk_id": f"{doc['metadata']['doc_id']}_chunk_{i}",
                    "checksum": checksum,
                    "timestamp": datetime.now().isoformat(),
                    "structured_data": json.dumps(doc["metadata"].get("structured_data", {}))
                }
                
                # Prepend title and authors to the chunk text to improve retrieval
                enriched_text = f"Title: {chunk_meta['title']}\nAuthors: {chunk_meta['authors']}\n\n{split}"
                chunks.append(Document(page_content=enriched_text, metadata=chunk_meta))
                
        print(f"Generated {len(chunks)} chunks.")
        return chunks

    def step5_6_embed_and_index(self, chunks, target="main"):
        """Step 5 & 6: Generate Embeddings and Vector DB Indexing"""
        print(f"\n[Step 5 & 6] Generating Embeddings and Indexing in {target.upper()} FAISS...")
        if not chunks:
            print("No chunks to index.")
            return
            
        # Update BM25 Corpus
        base = self.db_dir if target == "main" else self.user_db_dir
        corpus_path = os.path.join(base, "bm25_corpus.pkl")
        
        # Merge with existing corpus if extending (simplified: just overwrite or append logic needed)
        # For MVP: append to in-memory list and re-save full list
        
        current_corpus = self.bm25_corpus if target == "main" else self.user_bm25_corpus
        current_corpus.extend(chunks)
        
        # Save Corpus
        with open(corpus_path, 'wb') as f:
            pickle.dump(current_corpus, f)
            
        # Rebuild BM25 (in-memory)
        tokenized_corpus = [doc.page_content.lower().split() for doc in current_corpus]
        bm25_index = BM25Okapi(tokenized_corpus)
        
        if target == "main":
            self.bm25 = bm25_index
            self.bm25_corpus = current_corpus
        else:
            self.user_bm25 = bm25_index
            self.user_bm25_corpus = current_corpus

        if target == "user":
            if self.user_vector_store is None:
                self.user_vector_store = FAISS.from_documents(chunks, self.embeddings)
            else:
                self.user_vector_store.add_documents(chunks)
            # Save User Index
            self.user_vector_store.save_local(self.user_db_dir)
            print(f"Successfully indexed {len(chunks)} chunks and saved to {self.user_db_dir}")
            
        else: # Main Index
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            else:
                self.vector_store.add_documents(chunks)
            # Save Main Index
            self.vector_store.save_local(self.db_dir)
            print(f"Successfully indexed {len(chunks)} chunks and saved to {self.db_dir}")

    def step7_retrieve(self, query, top_k=5, filter_dict=None, source="main"):
        """Step 7: Retrieval Strategy (Hybrid: BM25 + Vector)"""
        # Determine store and BM25 index
        if source == "user":
             print(f"\n[Step 7] Retrieving top {top_k} from USER UPLOADS for query: '{query}'")
             vector_store = self.user_vector_store
             bm25_index = self.user_bm25
             corpus = self.user_bm25_corpus
        else:
             print(f"\n[Step 7] Retrieving top {top_k} from MAIN DATABASE for query: '{query}'")
             vector_store = self.vector_store
             bm25_index = self.bm25
             corpus = self.bm25_corpus

        if not vector_store:
             print(f"{source} vector store is empty.")
             return []

        # 1. Dense Retrieval (FAISS)
        # Fetch more candidates to fuse
        dense_k = top_k * 2
        try:
            dense_results = vector_store.similarity_search_with_score(query, k=min(dense_k, len(corpus) if corpus else dense_k))
        except Exception as e:
            print(f"Dense search failed: {e}")
            dense_results = []
            
        # 2. Sparse Retrieval (BM25)
        bm25_results = []
        if bm25_index and corpus:
            tokenized_query = query.lower().split()
            # Get top N scores
            doc_scores = bm25_index.get_scores(tokenized_query)
            top_indices = np.argsort(doc_scores)[::-1][:dense_k]
            
            # Normalize BM25 scores (0-1 approx, though BM25 is not bounded, we just use raw for fusion rank or simple norm)
            # Simple approach: Return candidate documents
            for idx in top_indices:
                if doc_scores[idx] > 0:
                    bm25_results.append((corpus[idx], float(doc_scores[idx])))
        
        # 3. Hybrid Fusion (Reciprocal Rank Fusion - RRF)
        # We combine the ranks from both systems
        
        # Map doc_id -> RRF score
        fused_scores = {}
        k_rrf = 60
        
        # Process Dense
        # Dense returns distance (lower is better for L2, but we need to check metric). 
        # FAISS default is L2. Lower is better.
        # But wait, similarity_search_with_score in LangChain uses L2 distance usually.
        # Let's assume list is already sorted by relevance (best first).
        for rank, (doc, score) in enumerate(dense_results):
            # doc_id unique key: use chunk_id if available or generate hash
            unique_id = doc.metadata.get("chunk_id") or doc.metadata.get("checksum") or doc.page_content[:50]
            if unique_id not in fused_scores:
                fused_scores[unique_id] = {"doc": doc, "score": 0.0}
            fused_scores[unique_id]["score"] += 1.0 / (k_rrf + rank + 1)
            fused_scores[unique_id]["source"] = "dense"

        # Process BM25
        # BM25 scores: higher is better. Sorted best first.
        for rank, (doc, score) in enumerate(bm25_results):
            unique_id = doc.metadata.get("chunk_id") or doc.metadata.get("checksum") or doc.page_content[:50]
            if unique_id not in fused_scores:
                fused_scores[unique_id] = {"doc": doc, "score": 0.0}
            fused_scores[unique_id]["score"] += 1.0 / (k_rrf + rank + 1)
            
        # Sort by Fused Score (High is good)
        sorted_candidates = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        
        # Format for output (Doc, score) 
        # Note: We return the fused score now, not the raw distance
        final_results = [(item["doc"], item["score"]) for item in sorted_candidates[:top_k]]
        
        print(f"Hybrid Fusion returned {len(final_results)} docs.")
        return final_results

    def _is_noisy_chunk(self, text):
        """Heuristic filter for reference-list like or low-quality OCR chunks."""
        if not text:
            return True

        t = str(text).strip()
        if len(t) < 120:
            return True

        lower = t.lower()
        reference_markers = [
            "references", "frontiers in", "doi:", "et al.", "journal of", "nature medicine",
            "new england journal", "proceedings of", "pmid", "issn"
        ]
        marker_hits = sum(1 for m in reference_markers if m in lower)

        # Many citation-number prefixes typically indicates bibliography block.
        citation_like = len(re.findall(r"(^|\s)\d{1,3}[\.)]", t)) >= 8

        # Dense OCR-merging artifact: very long alnum tokens without spaces.
        long_token = any(len(tok) > 35 for tok in re.findall(r"[A-Za-z0-9]+", t))

        return marker_hits >= 3 or citation_like or long_token

    def _clean_display_text(self, text, max_chars=700):
        if not text:
            return ""
        cleaned = re.sub(r"\s+", " ", str(text)).strip()
        # Remove hanging citation clusters like [12, 13, 14]
        cleaned = re.sub(r"\[(?:\d+[\s,;-]*){2,}\]", "", cleaned)
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars].rstrip() + " ..."
        return cleaned

    def step8_provenance(self, results, max_items=10):
        """Step 8: Provenance & Audit (Format results with evidence)"""
        formatted_results = []
        fallback_results = []
        
        # In a retrieval list, results is typically [(Document, score), ...]
        # We want to relax filtering to ensure we don't drop potential candidates unnecessarily
        
        for doc, score in results:
            meta = doc.metadata
            
            # Clean up title and ID display
            doc_id = meta.get('pmcid') or meta.get('pmid') or meta.get('doi') or "Docs"
            title = meta.get('title', 'Unknown Title')
            year = meta.get('year', 'n.d.')
            authors = meta.get('authors', '')
            
            # Remove HTML tags from title like <i>
            import html
            try:
                title = html.unescape(title)
                title = re.sub(r'<[^>]+>', '', title)
            except:
                pass

            author_str = f" by {authors}" if authors else ""
            evidence = f"[{doc_id}] {title}{author_str} ({year})"
            
            item = {
                "evidence": evidence,
                "text": self._clean_display_text(doc.page_content),
                "raw_text": doc.page_content,
                "score": score,
                "doi": meta.get("doi"),
                "structured_data": {}
            }

            # Recover structured_data if saved as JSON string in metadata
            sd = meta.get("structured_data")
            if isinstance(sd, str):
                try:
                    item["structured_data"] = json.loads(sd) if sd else {}
                except Exception:
                    item["structured_data"] = {}
            elif isinstance(sd, dict):
                item["structured_data"] = sd

            fallback_results.append(item)
            
            # Use a slightly less aggressive filter
            if not self._is_noisy_chunk(doc.page_content):
                formatted_results.append(item)

        # If strict filtering removed everything, fallback to raw results
        if not formatted_results:
            formatted_results = fallback_results

        # Sort by score
        # Heuristic: If scores are very small (<1.0) and typically look like RRF, or if we switched to normalized relevance, assumes Higher is Better.
        # FAISS L2 Distance: Lower is Better.
        # Check first element score
        if formatted_results:
            first_score = formatted_results[0].get("score", 0)
            # RRF scores are typically sum(1/rank), so < 1.0 usually. 
            # L2 distances for embeddings in high dim can be > 1.0 or < 1.0 depending on normalization.
            # Assuming Hybrid Search (step7) returns RRF (Higher is Better).
            formatted_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Return up to max_items
        return formatted_results[:max_items]

    def step9_incremental_update(self, max_results=5):
        """Step 9: Incremental Update & Monitoring (The Pipeline Orchestrator)"""
        print("==================================================")
        print("🚀 Starting Gene Editing Data Pipeline (9-Step MVP)")
        print("==================================================")
        
        query = self.step0_planning()
        metadata = self.step1_fetch_metadata(query, max_results=max_results)
        downloaded_files = self.step2_download_pdfs(metadata)
        
        # If no PDFs were downloaded, fallback to indexing abstracts
        if not downloaded_files:
            print("No PDFs downloaded. Falling back to indexing abstracts...")
            parsed_docs = [{"text": m["abstract"], "metadata": m} for m in metadata if m["abstract"]]
        else:
            parsed_docs = self.step3_parse_pdfs(downloaded_files)
            
        parsed_docs = self.step3_5_structured_extraction(parsed_docs)
        chunks = self.step4_chunking(parsed_docs)
        self.step5_6_embed_and_index(chunks)
        
        print("==================================================")
        print("✅ Pipeline Update Complete")
        print("==================================================")

if __name__ == "__main__":
    pipeline = GeneEditingDataPipeline()
    pipeline.step9_incremental_update(max_results=3)
    
    # Test Retrieval
    res = pipeline.step7_retrieve("What are the off-target effects of CRISPR-Cas9?")
    prov = pipeline.step8_provenance(res)
    for p in prov:
        print(f"\nEvidence: {p['evidence']}\nText: {p['text'][:200]}...\nScore: {p['score']}")
