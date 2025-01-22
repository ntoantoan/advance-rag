import os
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import logging

# Local imports
from splitter.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from vectordb.uploads import upload_pgvector
from utils import chat_completion_without_stream
from search.weight_rerank import WeightRerank
from cache_embedding import EmbeddingCache
from cleaner.text_extractor import TextExtractor
from cleaner.pdf_extractor import PdfExtractor
from cleaner.csv_extractor import CSVExtractor
from cleaner.docx_extractor import WordExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
STORAGE_PATH = "storage"
SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".txt", ".docx"}
EMBEDDING_DIMENSION = 1536

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str
    history: List[str]

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str

# Initialize services
redis_cache = EmbeddingCache()
weight_rerank = WeightRerank(redis_cache)

app = FastAPI(
    title="Advanced RAG API",
    description="API for Advanced RAG operations",
    version="1.0.0"
)

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint returning API status"""
    return {"status": "active", "message": "Welcome to Advanced RAG API"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload and process a file for RAG operations
    
    Args:
        file: Uploaded file object
    
    Returns:
        Dict containing upload status and metadata
    """
    try:
        # Validate file extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        # Create storage directory and save file
        os.makedirs(STORAGE_PATH, exist_ok=True)
        file_path = os.path.join(STORAGE_PATH, file.filename)
        
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        # Extract content based on file type
        extractors = {
            ".pdf": PdfExtractor,
            ".csv": CSVExtractor,
            ".txt": TextExtractor,
            ".docx": WordExtractor
        }
        
        extractor = extractors[file_extension](file_path)
        documents = extractor.extract()
        
        # Process and upload content
        contents = "\n".join(doc.page_content for doc in documents)
        upload_result = upload_pgvector(EMBEDDING_DIMENSION, contents)
        
        return upload_result

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process chat messages using RAG
    
    Args:
        request: ChatRequest object containing message and history
    
    Returns:
        ChatResponse object containing the response
    """
    try:
        documents = weight_rerank.run(request.message, k=5, hybird_search=True)
        response = chat_completion_without_stream(
            [{"role": "user", "content": request.message}],
            documents=documents,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        return ChatResponse(response=response)

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)