import redis
import numpy as np
import json
from typing import Optional, List

class EmbeddingCache:
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        """Initialize Redis connection for caching embeddings.
        
        Args:
            host: Redis host address
            port: Redis port number
            db: Redis database number
        """
        self.redis_client = redis.Redis(host=host, port=port, db=db)
        
    def store_embedding(self, key: str, embedding: List[float], ttl: Optional[int] = None) -> bool:
        """Store embedding vector in Redis cache.
        
        Args:
            key: Unique identifier for the embedding
            embedding: List containing the embedding vector
            ttl: Time-to-live in seconds (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert embedding list to JSON string
            embedding_json = json.dumps(embedding)
            
            # Store in Redis
            if ttl:
                success = self.redis_client.setex(key, ttl, embedding_json)
            else:
                success = self.redis_client.set(key, embedding_json)
            return bool(success)
        except Exception:
            return False
            
    def get_embedding(self, key: str) -> Optional[List[float]]:
        """Retrieve embedding vector from Redis cache.
        
        Args:
            key: Unique identifier for the embedding
            
        Returns:
            List[float] if found, None otherwise
        """
        try:
            result = self.redis_client.get(key)
            if result is None:
                return None
                
            return json.loads(result)
        except Exception:
            return None
            
    def delete_embedding(self, key: str) -> bool:
        """Delete embedding vector from Redis cache.
        
        Args:
            key: Unique identifier for the embedding
            
        Returns:
            bool: True if successful, False otherwise
        """
        return bool(self.redis_client.delete(key))

if __name__ == "__main__":
    # Example
    vector = [0.1, 0.2, 0.3] * 512  # Create a sample list of floats
    query = "What is the meaning of?"
    cache = EmbeddingCache()
    # cache.store_embedding(query, vector, ttl=60)
    print(cache.get_embedding(query))
