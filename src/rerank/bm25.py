import numpy as np
from typing import List, Dict
from collections import Counter

class BM25:
    """
    BM25 implementation for document reranking.
    
    Parameters:
    - k1: Term frequency saturation parameter (default: 1.5)
    - b: Length normalization parameter (default: 0.75)
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = None
        self.idf = None
        self.doc_lens = None
        self.avgdl = None
        self.total_docs = None
        
    def fit(self, documents: List[List[str]]):
        """
        Fit the BM25 model on a corpus of documents.
        
        Args:
            documents: List of tokenized documents where each document is a list of tokens
        """
        self.total_docs = len(documents)
        
        # Calculate document lengths and average document length
        self.doc_lens = [len(doc) for doc in documents]
        self.avgdl = sum(self.doc_lens) / self.total_docs
        
        # Calculate document frequencies for each term
        self.doc_freqs = Counter()
        for doc in documents:
            # Count each term only once per document
            self.doc_freqs.update(set(doc))
            
        # Calculate IDF scores
        self.idf = {}
        for term, doc_freq in self.doc_freqs.items():
            self.idf[term] = np.log((self.total_docs - doc_freq + 0.5) / 
                                  (doc_freq + 0.5) + 1.0)
    
    def score(self, query: List[str], document: List[str], doc_idx: int) -> float:
        """
        Calculate BM25 score for a query-document pair.
        
        Args:
            query: List of query tokens
            document: List of document tokens
            doc_idx: Index of the document in the fitted corpus
            
        Returns:
            float: BM25 score
        """
        score = 0.0
        doc_len = self.doc_lens[doc_idx]
        
        # Count term frequencies in document
        doc_term_freqs = Counter(document)
        
        for term in query:
            if term not in self.idf:
                continue
                
            # Calculate term frequency component
            tf = doc_term_freqs[term]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            
            score += self.idf[term] * numerator / denominator
            
        return score
    
    def rerank(self, query: List[str], documents: List[List[str]], 
               top_k: int = None) -> List[int]:
        """
        Rerank documents based on BM25 scores.
        
        Args:
            query: List of query tokens
            documents: List of tokenized documents
            top_k: Number of top documents to return (default: return all)
            
        Returns:
            List of document indices sorted by score in descending order
        """
        scores = []
        for idx, doc in enumerate(documents):
            score = self.score(query, doc, idx)
            scores.append((idx, score))
            
        # Sort by score in descending order
        ranked_docs = sorted(scores, key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            ranked_docs = ranked_docs[:top_k]
        
        #return index and score as a list
        return ranked_docs
        
# if __name__ == "__main__":
#     documents = [
#         ["this", "is", "a", "test"],
#         ["this", "is", "another", "test"],
#         ["this", "is", "a", "test", "again"]
#     ]
#     bm25 = BM25()
#     bm25.fit(documents)


#     print(bm25.rerank(["this", "is", "a", "test"], documents, top_k=5))