from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import tempfile
from pathlib import Path

# Import our modules
from document_parser import DocumentParser
from rag_system import RAGSystem

# Initialize FastAPI app
app = FastAPI(
    title="Bizz | Docs API",
    description="AI-powered document analysis and RAG system for business documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system instance
rag_system: Optional[RAGSystem] = None

# Pydantic models
class DocumentInfo(BaseModel):
    filename: str
    upload_time: str
    content_length: int
    chunks: int

class QueryRequest(BaseModel):
    question: str
    analysis_type: str = "general"

class QueryResponse(BaseModel):
    response: str
    analysis_type: str
    source_documents: List[Dict[str, str]]
    query_time: str
    total_documents: int

class SystemStatus(BaseModel):
    status: str
    rag_system_initialized: bool
    document_count: int
    model_type: Optional[str] = None
    embedding_model: Optional[str] = None
    vector_db: Optional[str] = None

# Initialize RAG system endpoint
@app.post("/initialize", response_model=Dict[str, str])
async def initialize_rag_system(
    model_type: str = Form("OpenAI GPT-4o mini"),
    embedding_model: str = Form("sentence-transformers/all-MiniLM-L6-v2"),
    vector_db: str = Form("ChromaDB (Local)")
):
    """Initialize the RAG system with specified configuration"""
    global rag_system
    
    try:
        # Set OpenAI API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=400, 
                detail="OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
            )
        
        rag_system = RAGSystem(
            model_type=model_type,
            embedding_model=embedding_model,
            vector_db=vector_db
        )
        
        return {"message": "RAG system initialized successfully", "status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing RAG system: {str(e)}")

# Upload document endpoint
@app.post("/upload", response_model=Dict[str, str])
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(
            status_code=400, 
            detail="RAG system not initialized. Please call /initialize first."
        )
    
    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.txt', '.md', '.markdown'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_extension}. Supported types: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Parse document
            parser = DocumentParser()
            text_content = parser.parse_document(tmp_path)
            
            # Add to RAG system
            rag_system.add_document(text_content, file.filename)
            
            return {
                "message": f"Document '{file.filename}' processed successfully",
                "filename": file.filename,
                "content_length": len(text_content),
                "status": "success"
            }
            
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
            
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing document: {str(e)}"
        )

# Query documents endpoint
@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system about uploaded documents"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(
            status_code=400, 
            detail="RAG system not initialized. Please call /initialize first."
        )
    
    try:
        response = rag_system.query(request.question, request.analysis_type)
        
        if 'error' in response:
            raise HTTPException(status_code=500, detail=response['error'])
        
        return QueryResponse(**response)
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error querying documents: {str(e)}"
        )

# Get system status endpoint
@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get current system status"""
    global rag_system
    
    if not rag_system:
        return SystemStatus(
            status="Not initialized",
            rag_system_initialized=False,
            document_count=0
        )
    
    return SystemStatus(
        status="Running",
        rag_system_initialized=True,
        document_count=rag_system.get_document_count(),
        model_type=rag_system.model_type,
        embedding_model=rag_system.embedding_model,
        vector_db=rag_system.vector_db
    )

# Get document info endpoint
@app.get("/documents", response_model=List[DocumentInfo])
async def get_documents():
    """Get information about all uploaded documents"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(
            status_code=400, 
            detail="RAG system not initialized. Please call /initialize first."
        )
    
    try:
        doc_info = rag_system.get_document_info()
        return [DocumentInfo(**info) for info in doc_info]
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error getting document info: {str(e)}"
        )

# Clear documents endpoint
@app.delete("/documents", response_model=Dict[str, str])
async def clear_documents():
    """Clear all uploaded documents"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(
            status_code=400, 
            detail="RAG system not initialized. Please call /initialize first."
        )
    
    try:
        rag_system.clear_documents()
        return {"message": "All documents cleared successfully", "status": "success"}
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error clearing documents: {str(e)}"
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Bizz | Docs API",
        "version": "1.0.0",
        "description": "AI-powered document analysis and RAG system",
        "endpoints": {
            "POST /initialize": "Initialize RAG system",
            "POST /upload": "Upload and process document",
            "POST /query": "Query documents with AI",
            "GET /status": "Get system status",
            "GET /documents": "Get document information",
            "DELETE /documents": "Clear all documents",
            "GET /health": "Health check"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
