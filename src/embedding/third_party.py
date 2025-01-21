from typing import List, Optional
import openai
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
import os
import asyncio

from dotenv import load_dotenv
load_dotenv()

class EmbeddingGenerator:
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        """
        Initialize the embedding generator with specified provider
        
        Args:
            provider: The embedding provider ("openai", "gemini", or "claude")
            api_key: API key for the selected provider
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self._setup_client()
    
    def _setup_client(self):
        """Setup the appropriate client based on the provider"""
        if self.provider == "openai":
            self.client = OpenAI(api_key=self.api_key)
        elif self.provider == "gemini":
            genai.configure(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for the input text
        
        Args:
            text: Input text to generate embedding for
            
        Returns:
            List of floating point numbers representing the embedding vector
        """
        if self.provider == "openai":
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=1536
            )
            return response.data[0].embedding
            
        elif self.provider == "gemini":
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text
            )
            return result["embedding"]
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        return [self.get_embedding(text) for text in texts]
    

# if __name__ == "__main__":
#     test_openai = EmbeddingGenerator(provider="openai", api_key=os.getenv("OPENAI_API_KEY"))
#     embedding = test_openai.get_embedding("Hello, world!")
#     print(len(embedding))

#     test_gemini = EmbeddingGenerator(provider="gemini", api_key=os.getenv("GEMINI_API_KEY"))
#     embedding = test_gemini.get_embedding("Hello, world!")
#     print(len(embedding))

#     test_batch_openai = EmbeddingGenerator(provider="openai", api_key=os.getenv("OPENAI_API_KEY"))
#     embedding = test_batch_openai.get_batch_embeddings(["Hello, world!", "Hello, world!", "Hello, world!"])
#     print(len(embedding))

#     test_batch_gemini = EmbeddingGenerator(provider="gemini", api_key=os.getenv("GEMINI_API_KEY"))
#     embedding = test_batch_gemini.get_batch_embeddings(["Hello, world!", "Hello, world!", "Hello, world!"])
#     print(len(embedding))

