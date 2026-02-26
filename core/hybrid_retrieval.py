import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
# In a real scenario, you would use a proper embedding model like text-embedding-3-small
# Here we use a dummy embedding for demonstration purposes.

class HybridRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.tokenized_corpus = [doc.lower().split(" ") for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Dummy embeddings for demonstration
        self.document_embeddings = np.random.rand(len(documents), 128)

    def _get_dummy_embedding(self, text):
        return np.random.rand(1, 128)

    def semantic_search(self, query, top_k=5):
        query_embedding = self._get_dummy_embedding(query)
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        
        # Min-Max Normalize
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)
        if max_sim > min_sim:
            normalized_sims = (similarities - min_sim) / (max_sim - min_sim)
        else:
            normalized_sims = similarities
            
        top_indices = np.argsort(normalized_sims)[::-1][:top_k]
        return top_indices, normalized_sims

    def lexical_search(self, query, top_k=5):
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Min-Max Normalize
        min_score = np.min(bm25_scores)
        max_score = np.max(bm25_scores)
        if max_score > min_score:
            normalized_scores = (bm25_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = bm25_scores
            
        top_indices = np.argsort(normalized_scores)[::-1][:top_k]
        return top_indices, normalized_scores

    def hybrid_search(self, query, alpha=0.5, top_k=5):
        """
        Combines semantic and lexical search scores.
        alpha: weight for lexical search (BM25). (1-alpha) is weight for semantic search.
        """
        _, semantic_scores = self.semantic_search(query, top_k=len(self.documents))
        _, lexical_scores = self.lexical_search(query, top_k=len(self.documents))

        hybrid_scores = alpha * lexical_scores + (1 - alpha) * semantic_scores
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "score": hybrid_scores[idx]
            })
        return results

if __name__ == "__main__":
    docs = [
        "CRISPR-Cas9 is used for DNA double-strand breaks.",
        "Base editing allows for precise point mutation repair without double-strand breaks.",
        "Prime editing can insert small fragments of DNA.",
        "FDA approved PARP inhibitors for BRCA1/2 mutated ovarian cancer.",
        "Osimertinib is an EGFR TKI for non-small cell lung cancer."
    ]
    retriever = HybridRetriever(docs)
    results = retriever.hybrid_search("What is approved for BRCA mutation?", alpha=0.6)
    print("Hybrid Search Results:")
    for res in results:
        print(f"Score: {res['score']:.4f} | Doc: {res['document']}")
