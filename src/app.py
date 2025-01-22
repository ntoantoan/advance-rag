import os
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from splitter.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from vectordb.uploads import upload_milvus, upload_pgvector
from utils import chat_completion_without_stream
from search.weight_rerank import WeightRerank
from cache_embedding import EmbeddingCache
from cleaner.text_extractor import TextExtractor                                                        
from cleaner.pdf_extractor import PdfExtractor
from cleaner.csv_extractor import CSVExtractor
from cleaner.docx_extractor import WordExtractor
storage_path = "storage"


redis_cache = EmbeddingCache()
weight_rerank = WeightRerank(redis_cache)

# Initialize FastAPI app
app = FastAPI(
    title="Advance RAG API",
    description="API for Advanced RAG operations",
    version="1.0.0"
)

documents_db = []
@app.get("/")
async def root():
    """Root endpoint returning API status"""
    return {"status": "active", "message": "Welcome to Advance RAG API"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Create storage directory if it doesn't exist
    os.makedirs(storage_path, exist_ok=True)
    
    contents = await file.read()
    with open(f"{storage_path}/{file.filename}", "wb") as f:
        f.write(contents)
    
    if file.filename.endswith(".pdf"):
        extractor = PdfExtractor(f"{storage_path}/{file.filename}")
        documents = extractor.extract()
    elif file.filename.endswith(".csv"):
        extractor = CSVExtractor(f"{storage_path}/{file.filename}")
        documents = extractor.extract()

    elif file.filename.endswith(".txt"):
        extractor = TextExtractor(f"{storage_path}/{file.filename}")
        documents = extractor.extract()
    elif file.filename.endswith(".docx"):
        extractor = WordExtractor(f"{storage_path}/{file.filename}")
        documents = extractor.extract()
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    

    contents = [doc.page_content for doc in documents]

    #to string
    contents = "\n".join(contents)
    # contents = contents.decode('utf-8')  # Decode bytes to string
    # upload_milvus(collection_name="rag_collection", dim=1536, contents=contents)
    upload_pg_result = upload_pgvector(1536, contents)
    return upload_pg_result


class ChatRequest(BaseModel):
    message: str
    history: List[str]

@app.post("/chat")
async def chat(request: ChatRequest):
    query = request.message
    documents = weight_rerank.run(query, k=5, hybird_search=True)
    response = chat_completion_without_stream([{"role": "user", "content": request.message}], documents=documents, api_key=os.getenv("OPENAI_API_KEY"))
    return {"response": response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)