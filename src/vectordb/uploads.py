import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File
import os

from vectordb.milvus_vectordb import MilvusVectorDB
from splitter.text_splitter import RecursiveCharacterTextSplitter
from embedding.third_party import EmbeddingGenerator
from vectordb.pgvector import PGVector

def _process_text_to_embeddings(contents: str) -> tuple[List[str], List[List[float]]]:
    """
    Helper function to process text into chunks and generate embeddings.
    
    Args:
        contents: Input text to process
        
    Returns:
        Tuple of (chunks, embeddings)
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(contents)
    
    embedding_generator = EmbeddingGenerator(
        provider="openai",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    embeddings = embedding_generator.get_batch_embeddings(chunks)

    if len(chunks) != len(embeddings):
        raise HTTPException(
            status_code=500,
            detail=f"Mismatch in number of chunks ({len(chunks)}) and embeddings ({len(embeddings)})"
        )
        
    return chunks, embeddings

def upload_milvus(collection_name: str, dim: int, contents: str) -> Dict[str, str]:
    """
    Upload text content to Milvus vector database.
    
    Args:
        collection_name: Name of the Milvus collection
        dim: Dimension of the vectors
        contents: Text content to process and upload
        
    Returns:
        Status dictionary
    """
    vdb = MilvusVectorDB(collection_name=collection_name, dim=dim)
    chunks, embeddings = _process_text_to_embeddings(contents)
    vdb.add_documents(chunks, embeddings)
    return {"status": "success"}

def upload_pgvector(dim: int, contents: str) -> Dict[str, str]:
    """
    Upload text content to PGVector database.
    
    Args:
        dim: Dimension of the vectors
        contents: Text content to process and upload
        batch_size: Size of batches for processing
        
    Returns:
        Status dictionary
    """
    vdb = PGVector(connection_string="postgresql://postgres:postgres@localhost:5432/vectordb")
    chunks, embeddings = _process_text_to_embeddings(contents)
    vdb.add_vectors(embeddings, chunks)
    return {"status": "success"}

