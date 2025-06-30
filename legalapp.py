## 3. Updated legalapp.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from typing import List
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from legal_rag_system import LegalAssistant

app = FastAPI(
    title="Legal Document QA System",
    description="API for querying legal documents using RAG pipeline",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our assistant
assistant = LegalAssistant()
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QuestionRequest(BaseModel):
    question: str
    k: int = 3  # Number of results to return

class DocumentUploadResponse(BaseModel):
    message: str
    filenames: List[str]
    total_documents: int

class AnswerResponse(BaseModel):
    answer: str
    sources: List[dict]

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload and process legal documents (PDFs or text files).
    The system will extract text, create embeddings, and build a search index.
    """
    saved_files = []
    for file in files:
        try:
            # Validate file type
            if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.filename}. Only PDF and TXT files are supported."
                )
            
            # Save the uploaded file
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing {file.filename}: {str(e)}"
            )
    
    # Load and index all documents (existing + new)
    try:
        all_files = [
            os.path.join(UPLOAD_DIR, f) 
            for f in os.listdir(UPLOAD_DIR) 
            if f.endswith(('.pdf', '.txt'))
        ]
        assistant.load_documents(all_files, persist_dir="legal_index")
        
        return DocumentUploadResponse(
            message=f"Successfully processed {len(saved_files)} document(s)",
            filenames=[os.path.basename(f) for f in saved_files],
            total_documents=len(all_files)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading documents: {str(e)}"
        )

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the uploaded legal documents.
    The system will return an answer along with references to the source documents.
    """
    try:
        if not os.listdir(UPLOAD_DIR):
            raise HTTPException(
                status_code=400,
                detail="No documents have been uploaded yet. Please upload documents first."
            )
            
        result = assistant.ask_question(request.question, k=request.k)
        return AnswerResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/health")
async def health_check():
    """Check if the API is running and ready"""
    return {
        "status": "healthy",
        "loaded_documents": len(os.listdir(UPLOAD_DIR)) if os.path.exists(UPLOAD_DIR) else 0
    }

@app.get("/reset")
async def reset_system():
    """Clear all uploaded documents and reset the system"""
    try:
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            os.unlink(file_path)
        return {"message": "System reset successfully", "documents_remaining": 0}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error resetting system: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "legalapp:app",  # <-- pass as import string
        host="0.0.0.0",
        port=8001,
        reload=True
    )