from typing import List
from rerank.bm25 import BM25

def convert_documents_to_bm25(documents: List[str]) -> List[List[str]]:
    return [document.split() for document in documents]

def convert_query_to_bm25(query: str) -> List[str]:
    query_tokens = query.split()
    return query_tokens

class WeightRerank:
    def __init__(self):
        self.bm25 = BM25()

    def run(self, query: List[str], documents: List[str], k: int = 5) -> List[str]:
        bm25_corpus = convert_documents_to_bm25(documents)
        self.bm25.fit(bm25_corpus)
        scores = self.bm25.rerank(query, documents, top_k=k)
        return scores



if __name__ == "__main__":
    wr = WeightRerank()
    documents = [
        "this is a test",
        "this is another test",
        "this is a test again"
    ]   
    query = "this is a test"
    query_tokens = convert_query_to_bm25(query)
    score = wr.run(query_tokens, documents, k=5)
    print(score)