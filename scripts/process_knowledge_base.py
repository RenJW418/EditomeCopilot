"""
process_knowledge_base.py — Build FAISS + BM25 index from literature JSON DB.

Strategy: ONE ARTICLE = ONE CHUNK.
Scientific abstracts are already coherent semantic units (~200-350 words).
Splitting them destroys context and hurts retrieval quality.

Statistics from 86K-article DB:
  - Median abstract: 1,429 chars, P95: 2,357, P99: 2,983
  - Only 5.3% of abstracts are ≤800 chars (the old chunk_size)
  - Splitting at 800 chars fragments 94.7% of abstracts — terrible.

Benefits of one-article-per-chunk:
  - ~86K chunks (vs 377K with splitting) → 4× faster build
  - Better embedding quality: full abstract context
  - Better retrieval: no orphan fragments
  - Better reranking: reranker sees complete abstracts
"""

import json
import os
import pickle
import time
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# Max chars per chunk — safety truncation for rare outliers.
# P99 of abstracts is ~3000 chars; 512-token PubMedBERT ≈ 2500-3000 chars.
# We set 3000 as soft limit to preserve most abstracts intact.
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "3000"))


class KnowledgeBaseProcessor:
    def __init__(self, data_file, db_dir="data/faiss_db"):
        self.data_file = data_file
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)

        # Initialize Embeddings — use same priority chain as core/data_pipeline.py
        # to guarantee vector space consistency across all index builders.
        print("Initializing Embedding Model...")
        _default_model = os.getenv(
            "EMBEDDING_MODEL",
            "pritamdeka/S-PubMedBert-MS-MARCO",  # Best biomedical retrieval model
        )
        _fallbacks = [
            _default_model,
            "pritamdeka/S-PubMedBert-MS-MARCO",
            "dmis-lab/biobert-base-cased-v1.2",
            "sentence-transformers/all-MiniLM-L6-v2",
        ]
        self.embeddings = None
        for model_name in _fallbacks:
            try:
                print(f"  Trying model: {model_name}")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": os.getenv("EMBED_DEVICE", "cpu")},
                    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
                )
                print(f"  Loaded: {model_name}")
                break
            except Exception as e:
                print(f"  Failed to load {model_name}: {e}")
        if self.embeddings is None:
            raise RuntimeError("No embedding model could be loaded.")

        print(f"  Strategy: ONE ARTICLE = ONE CHUNK (max {MAX_CHUNK_CHARS} chars)")

    def load_data(self):
        print(f"Loading data from {self.data_file}...")
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return []

    def create_documents(self, raw_data):
        """Convert raw JSON articles to LangChain Documents.
        
        Each article becomes ONE Document. The page_content includes structured
        fields (title, authors, journal, year, abstract) so embeddings capture
        richer context for retrieval.
        
        For the rare cases where abstract > MAX_CHUNK_CHARS, we truncate at a
        sentence boundary to stay within the embedding model's effective range.
        """
        documents = []
        print(f"Processing {len(raw_data)} articles...")

        stats = {"total": len(raw_data), "no_abstract": 0, "truncated": 0, "skipped": 0}

        for article in raw_data:
            title = (article.get("title") or "").strip()
            abstract = (article.get("abstract") or "").strip()
            year = article.get("year", "Unknown")
            source = article.get("source", "")
            authors = article.get("authors", "")
            journal = article.get("journal", "")
            doi = article.get("doi", "")

            # Build structured text block
            parts = []
            parts.append(f"TITLE: {title}" if title else "TITLE: Untitled")

            if authors:
                # Compress author list: first 3 authors + "et al."
                if isinstance(authors, list):
                    authors = ", ".join(authors)
                author_short = authors[:200] + ("..." if len(authors) > 200 else "")
                parts.append(f"AUTHORS: {author_short}")

            if journal:
                parts.append(f"JOURNAL: {journal}")

            if year and year != "Unknown":
                parts.append(f"YEAR: {year}")

            if abstract and len(abstract) >= 10:
                # Truncate extremely long abstracts at sentence boundary
                if len(abstract) > MAX_CHUNK_CHARS:
                    cut = abstract[:MAX_CHUNK_CHARS]
                    # Try to cut at last sentence boundary
                    last_period = cut.rfind(". ")
                    if last_period > MAX_CHUNK_CHARS * 0.6:
                        abstract = cut[:last_period + 1]
                    else:
                        abstract = cut + "..."
                    stats["truncated"] += 1
                parts.append(f"ABSTRACT: {abstract}")
            else:
                stats["no_abstract"] += 1
                # Title-only entries: keep only if title is meaningful
                if not title or len(title) < 20:
                    stats["skipped"] += 1
                    continue

            content = "\n".join(parts)

            # Metadata for filtering/display (not embedded, just stored)
            metadata = {
                "gea_id": article.get("gea_id", ""),
                "source": source,
                "url": article.get("url", ""),
                "doi": doi,
                "year": year,
                "title": title[:150],
                "journal": journal[:100] if journal else "",
            }

            documents.append(Document(page_content=content, metadata=metadata))

        print(f"  > Created {len(documents):,} documents (1 article = 1 chunk)")
        print(f"  > No abstract (title-only): {stats['no_abstract']:,}")
        print(f"  > Truncated (>{MAX_CHUNK_CHARS} chars): {stats['truncated']:,}")
        print(f"  > Skipped (no useful content): {stats['skipped']:,}")
        return documents

    def build_index(self, chunks):
        """Build FAISS vector index + BM25 sparse index with progress bar."""
        total = len(chunks)
        batch_size = 500  # embed 500 chunks at a time
        print(f"\n{'='*60}")
        print(f"Building FAISS Index: {total:,} chunks")
        print(f"Batch size: {batch_size} | Model: {self.embeddings.model_name}")
        print(f"{'='*60}")
        start_time = time.time()

        vector_store = None

        for i in tqdm(range(0, total, batch_size), desc="Embedding", unit="batch",
                       total=(total + batch_size - 1) // batch_size):
            batch = chunks[i : i + batch_size]
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, self.embeddings)
            else:
                batch_store = FAISS.from_documents(batch, self.embeddings)
                vector_store.merge_from(batch_store)

        elapsed = time.time() - start_time
        speed = total / elapsed if elapsed > 0 else 0
        print(f"\n  FAISS index built in {elapsed:.1f}s ({speed:.0f} chunks/s)")

        # Save FAISS
        print(f"  Saving FAISS index to {self.db_dir}...")
        vector_store.save_local(self.db_dir)

        # Build BM25 sparse index
        print("  Building BM25 index...")
        tokenized = [doc.page_content.lower().split() for doc in tqdm(chunks, desc="Tokenizing", unit="doc")]
        bm25 = BM25Okapi(tokenized)
        bm25_path = os.path.join(self.db_dir, "bm25_corpus.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump(chunks, f)
        print(f"  BM25 corpus saved ({len(chunks):,} docs) → {bm25_path}")

        print(f"\n  All done. Total time: {time.time() - start_time:.1f}s")

    def run(self):
        data = self.load_data()
        if not data:
            return

        docs = self.create_documents(data)
        # No splitting — one article = one chunk
        if docs:
            self.build_index(docs)

if __name__ == "__main__":
    # Adjust filename if needed based on version q1
    input_file = "data/knowledge_base/literature_db_GEA_v2026_Q1.json" 
    
    # Check if file exists, else try to find latest
    if not os.path.exists(input_file):
        import glob
        files = glob.glob("data/knowledge_base/literature_db_GEA_v*.json")
        if files:
           files.sort()
           input_file = files[-1]
           print(f"Auto-selected latest file: {input_file}")
    
    if os.path.exists(input_file):
        processor = KnowledgeBaseProcessor(input_file)
        processor.run()
    else:
        print(f"Input file not found: {input_file}")
