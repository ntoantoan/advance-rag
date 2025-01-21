import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
import os

from vectordb.milvus_vectordb import MilvusVectorDB
from splitter.text_splitter import RecursiveCharacterTextSplitter
from embedding.third_party import EmbeddingGenerator
from vectordb.pgvector import PGVector


def upload_milvus(collection_name: str, dim: int, contents: str):
    vdb = MilvusVectorDB(collection_name=collection_name, dim=dim)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(contents)
    # Get embeddings for chunks
    embedding_generator = EmbeddingGenerator(
        provider="openai", 
        api_key=os.getenv("OPENAI_API_KEY")
    )
    chunks_to_process = chunks 
    embeddings = embedding_generator.get_batch_embeddings(chunks_to_process)

    # Verify the lengths match
    if len(chunks_to_process) != len(embeddings):
        raise HTTPException(
            status_code=500,
            detail=f"Mismatch in number of chunks ({len(chunks_to_process)}) and embeddings ({len(embeddings)})"
        )

    vdb.add_documents(chunks_to_process, embeddings)
    return {"status": "success"}


def upload_pgvector(dim: int, contents: str, batch_size: int = 2):
    vdb = PGVector(connection_string="postgresql://postgres:postgres@localhost:5432/vectordb")
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
    vdb.add_vectors(embeddings, chunks)
    return {"status": "success"}

