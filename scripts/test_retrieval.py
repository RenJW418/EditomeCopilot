import sys
import os
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_knowledge_base():
    db_dir = "data/faiss_db"
    
    print("Loading Embedding Model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print(f"Loading FAISS Index from {db_dir}...")
    try:
        vector_store = FAISS.load_local(db_dir, embeddings, allow_dangerous_deserialization=True)
        print(f"Index loaded. Total vectors: {vector_store.index.ntotal}")
    except Exception as e:
        print(f"Error loading index: {e}")
        return

    # Test Queries
    queries = [
        "latest prime editing efficiency improvements 2024",
        "CRISPR off-target detection methods methods",
        "Jennifer Doudna Cas9 review",
        "base editing in plants"
    ]

    print("\n=== Retrieval Test ===")
    for query in queries:
        print(f"\nQUERY: {query}")
        start = time.time()
        docs = vector_store.similarity_search(query, k=3)
        duration = time.time() - start
        
        print(f"  > Retrieved 3 docs in {duration:.4f}s")
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            year = doc.metadata.get("year", "N/A")
            title = doc.metadata.get("title", "No Title")
            print(f"    [{i+1}] {title[:60]}... ({year}) - {source}")
            # print(f"        Snippet: {doc.page_content[:100]}...")

if __name__ == "__main__":
    test_knowledge_base()
