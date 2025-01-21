import os
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from splitter.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from vectordb.uploads import upload_milvus, upload_pgvector
from utils import chat_completion_without_stream
from search.weight_rerank import WeightRerank

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
    contents = await file.read()
    # contents = contents.decode('utf-8')  # Decode bytes to string
    # upload_milvus(collection_name="rag_collection", dim=1536, contents=contents)
    upload_pg = upload_pgvector(1536, contents)
    print(upload_pg)
    return {
        "filename": file.filename, 
        "status": "success"
    }

class ChatRequest(BaseModel):
    message: str
    history: List[str]

@app.post("/chat")
async def chat(request: ChatRequest):
    query = request.message
    documents = WeightRerank().run(query, k=5, hybird_search=True)
    response = chat_completion_without_stream([{"role": "user", "content": request.message}], documents=documents, api_key=os.getenv("OPENAI_API_KEY"))
    return {"response": response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)