import json
import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

class KnowledgeBaseProcessor:
    def __init__(self, data_file, db_dir="data/faiss_db"):
        self.data_file = data_file
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Initialize Embeddings
        print("Initializing Embedding Model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize Splitter
        # Chunk size 1000 with overlap 200 is good for RAG tasks on scientific abstracts
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

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
        documents = []
        print(f"Processing {len(raw_data)} articles...")
        
        count_missing_abstract = 0
        
        for article in raw_data:
            # Combine Title and Abstract for better context
            title = article.get("title", "") or "No Title"
            abstract = article.get("abstract", "")
            
            # Skip if no abstract (or maybe just index title?)
            # Valid Abstract is crucial for RAG quality.
            if not abstract or len(abstract) < 10:
                count_missing_abstract += 1
                if len(title) > 20: 
                     # Fallback: Index title only if meaningful
                     content = f"TITLE: {title}\nYEAR: {article.get('year', 'Unknown')}\nSOURCE: {article.get('source', '')}"
                else:
                     continue
            else:
                content = f"TITLE: {title}\nABSTRACT: {abstract}\nYEAR: {article.get('year', 'Unknown')}\nSOURCE: {article.get('source', '')}"

            # Metadata for retrieval context
            metadata = {
                "geo_id": article.get("gea_id"),
                "source": article.get("source"),
                "url": article.get("url"),
                "doi": article.get("doi"),
                "year": article.get("year"),
                "title": title[:100] # Truncate title in metadata to save space
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
            
        print(f"  > Skipped/Title-only for {count_missing_abstract} articles with missing abstracts.")
        return documents

    def split_documents(self, documents):
        print(f"Splitting {len(documents)} documents...")
        chunks = self.splitter.split_documents(documents)
        print(f"  > Generated {len(chunks)} chunks.")
        return chunks

    def build_index(self, chunks):
        print("Building FAISS Index (this may take a while)...")
        start_time = time.time()
        
        # Batch processing for FAISS to manage memory if needed
        # But for ~100k chunks, all-MiniLM fits in memory easily.
        
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        end_time = time.time()
        print(f"  > Index building completed in {end_time - start_time:.2f} seconds.")
        
        # Save to disk
        print(f"Saving index to {self.db_dir}...")
        vector_store.save_local(self.db_dir)
        print("Done.")

    def run(self):
        data = self.load_data()
        if not data:
            return
        
        docs = self.create_documents(data)
        chunks = self.split_documents(docs)
        if chunks:
            self.build_index(chunks)

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
