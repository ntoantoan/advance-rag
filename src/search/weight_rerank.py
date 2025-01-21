import os
from typing import List
from search.bm25 import BM25
from embedding.third_party import EmbeddingGenerator
from vectordb.pgvector import PGVector


def convert_documents_to_bm25(documents: List[str]) -> List[List[str]]:
    return [document.split() for document in documents]

def convert_query_to_bm25(query: str) -> List[str]:
    query_tokens = query.split()
    return query_tokens

class WeightRerank:
    def __init__(self, ):
        self.bm25 = BM25()
        self.embedding_generator = EmbeddingGenerator(provider="openai", api_key=os.getenv("OPENAI_API_KEY"))
        self.pgvector = PGVector(connection_string="postgresql://postgres:postgres@localhost:5432/vectordb")


    def get_query_vector(self, query: List[str]) -> List[float]:
        return self.embedding_generator.get_embedding(query)
    
    def get_ranking_vectordb(self, embedding: List[float], k: int = 5) -> List[str]:
        return self.pgvector.search_vectors(embedding, k=k)
    
    def get_output_documents(self, query_scores: List[str], ranking_vectordb: List[str]) -> List[str]:
        pass


    def run(self, query: str, k: int = 5, hybird_search: bool = False, vector_search: bool = False) -> List[str]:
        """
        Implement the weight reranking algorithm
        score = 0.4/(1+score_bm25) + 0.6/(1+score_vectordb)
        """

        document_contents = []
        rerank_documents = []
        # Convert query string to tokens for BM25
        query_tokens = convert_query_to_bm25(query)
        
        # Get initial documents from full text search
        # documents = self.pgvector.full_text_search(query, k=50)
        all_documents = self.pgvector.get_all_vectors()

        if len(all_documents) == 0:
            print("No documents found by full text search")
            return []

        # Process documents for BM25
        if hybird_search:  
            contents = [doc.content for doc in all_documents]
            bm25_corpus = convert_documents_to_bm25(contents)
            self.bm25.fit(bm25_corpus)
            bm25_scores = self.bm25.rerank(query_tokens, bm25_corpus, top_k=10)

            query_scores = [score for _, score in bm25_scores][:k]
            document_contents = [contents[idx] for idx, _ in bm25_scores][:10]


            query_vector = self.get_query_vector(query)
            vector_scores = self.get_ranking_vectordb(query_vector, k=10)

            document_vectors_store = [doc.content for doc in all_documents]

            pair_scores = []
            for document, document_vs, query_score, vector_score in zip(document_contents, document_vectors_store, query_scores, vector_scores):
                score = 0.4/(1+query_score) + 0.6/(1+vector_score["similarity"])
                pair_scores.append((document, score))

            pair_scores.sort(key=lambda x: x[1], reverse=True)
            rerank_documents = [document for document, _ in pair_scores[:k]]

            return rerank_documents

        elif vector_search:
            query_vector = self.get_query_vector(query)
            vector_scores = self.get_ranking_vectordb(query_vector, k=k)
            # rerank_documents = [contents[idx] for idx, _ in vector_scores]
            rerank_documents = [doc["content"] for doc in vector_scores]
            print(rerank_documents)
        
        return rerank_documents


if __name__ == "__main__":
    wr = WeightRerank()
    # documents = [
    #     "this is a test",
    #     "this is another test",
    #     "this is a test again"
    # ]   
    # query = "this is a test"
    # query_tokens = convert_query_to_bm25(query)
    # score = wr.run(query_tokens, documents, k=5)
    # print(score)
    docs = wr.run("queries and documents", k=5, vector_search=True)
    print(docs)
