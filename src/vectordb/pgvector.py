import json
from typing import List, Optional
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

class PGVector:
    def __init__(self, connection_string: str):
        """Initialize PGVector with database connection string.
        
        Args:
            connection_string: PostgreSQL connection string
        """
        self.engine = create_engine(connection_string)
        self._init_db()

    def _init_db(self):
        """Initialize the database with required extensions and tables."""
        with Session(self.engine) as session:
            # Enable pgvector extension
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            
            # Create table for storing vectors
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS vector_store (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    embedding vector(1536),
                    metadata JSONB
                );
            """))
            session.commit()

    def add_vectors(self, vectors: List[np.ndarray], contents: List[str], metadata: Optional[List[dict]] = None):
        """Add vectors to the database.
        
        Args:
            vectors: List of numpy arrays representing embeddings
            contents: List of content strings associated with vectors
            metadata: Optional list of metadata dictionaries
        """
        if metadata is None:
            metadata = [{}] * len(vectors)
            
        with Session(self.engine) as session:
            for vector, content, meta in zip(vectors, contents, metadata):
                vector_str = f"[{','.join(map(str, vector))}]"
                session.execute(text("""
                    INSERT INTO vector_store (content, embedding, metadata)
                    VALUES (:content, :embedding, :metadata);
                """), {
                    "content": content,
                    "embedding": vector_str,
                    "metadata": json.dumps(meta)  # Convert dict to string for JSON parsing
                })
            session.commit()

    def search_vectors(self, query_vector: np.ndarray, k: int = 5) -> List[dict]:
        """Search for similar vectors using cosine similarity.
        
        Args:
            query_vector: Query vector to search for
            k: Number of results to return
            
        Returns:
            List of dictionaries containing search results
        """
        query_vector_str = f"[{','.join(map(str, query_vector))}]"
        
        with Session(self.engine) as session:
            results = session.execute(text("""
                SELECT content, metadata, 
                       1 - (embedding <=> :query_embedding) as similarity
                FROM vector_store
                ORDER BY embedding <=> :query_embedding
                LIMIT :k;
            """), {
                "query_embedding": query_vector_str,
                "k": k
            })
            
            return [
                {
                    "content": row[0],
                    "metadata": row[1],
                    "similarity": float(row[2])
                }
                for row in results
            ]
    def full_text_search(self, query: str, k: int = 5) -> List[dict]:
        with Session(self.engine) as session:
            results = session.execute(text("SELECT * FROM vector_store WHERE content ILIKE :query LIMIT :k"), {
                "query": f"%{query}%",
                "k": k
            })
            return [row for row in results]
    
    def get_all_vectors(self) -> List[dict]:
        with Session(self.engine) as session:
            results = session.execute(text("SELECT * FROM vector_store"))
            return [row for row in results]

if __name__ == "__main__":
    vdb = PGVector(connection_string="postgresql://postgres:postgres@localhost:5432/vectordb")
    vdb.add_vectors([np.random.rand(1536).tolist()], ["test"])
    print(vdb.search_vectors(np.random.rand(1536).tolist(), k=5))

    print(vdb.full_text_search("test", k=5))