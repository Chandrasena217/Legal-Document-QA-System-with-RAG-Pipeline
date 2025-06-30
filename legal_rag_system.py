# filepath: legal-rag-system/legal-rag-system/legal_rag_system.py
import os
import fitz  # PyMuPDF
import re
import pickle
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader

# Embeddings and vector store
from sentence_transformers import SentenceTransformer
import faiss

# LLM imports
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class DocumentProcessor:
    """Handles loading and processing of legal documents in various formats"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len,
            add_start_index=True
        )
    
    def load_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Load a document based on its file type"""
        if file_path.endswith('.pdf'):
            return self._load_pdf(file_path)
        elif file_path.endswith('.txt'):
            return self._load_text(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def _load_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Load and extract text from PDF files"""
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            documents = []
            
            for page in pages:
                # Clean the text
                text = self._clean_text(page.page_content)
                metadata = {
                    "source": file_path,
                    "page": page.metadata["page"],
                    "total_pages": len(pages)
                }
                documents.append({"text": text, "metadata": metadata})
            
            return documents
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
            return []
    
    def _load_text(self, file_path: str) -> List[Dict[str, Any]]:
        """Load and process plain text files"""
        try:
            loader = TextLoader(file_path)
            documents = loader.load()
            
            processed_docs = []
            for doc in documents:
                text = self._clean_text(doc.page_content)
                metadata = {"source": file_path}
                processed_docs.append({"text": text, "metadata": metadata})
            
            return processed_docs
        except Exception as e:
            print(f"Error loading text file {file_path}: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove special characters except those relevant for legal docs
        text = re.sub(r'[^\w\s.,;:()\'\"\-§¶]', '', text)
        return text
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into chunks for processing"""
        chunked_docs = []
        
        for doc in documents:
            text = doc["text"]
            metadata = doc["metadata"]
            
            # Split the text into chunks
            chunks = self.text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = i
                chunked_docs.append({
                    "text": chunk,
                    "metadata": chunk_metadata
                })
        
        return chunked_docs

class VectorStore:
    """Handles embedding generation and vector similarity search"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.dimension = 384  # Dimension of all-MiniLM-L6-v2 embeddings
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        return self.model.encode(texts, show_progress_bar=True, batch_size=32)
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """Build FAISS index from documents"""
        if not documents:
            raise ValueError("No documents provided to index")
            
        texts = [doc["text"] for doc in documents]
        embeddings = self.generate_embeddings(texts)
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        self.documents = documents
        
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents to the query"""
        if self.index is None:
            raise ValueError("Index has not been built")
            
        query_embedding = self.generate_embeddings([query])
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:  # FAISS returns -1 for invalid indices
                doc = self.documents[idx]
                # Convert distance to similarity score (higher is better)
                score = 1 / (1 + distances[0][i])
                results.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": float(score)
                })
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    
    def save_index(self, save_path: str):
        """Save the index and documents to disk"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(save_path, "index.faiss"))
        
        # Save documents
        with open(os.path.join(save_path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
    
    def load_index(self, load_path: str):
        """Load index and documents from disk"""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Directory {load_path} does not exist")
            
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(load_path, "index.faiss"))
        
        # Load documents
        with open(os.path.join(load_path, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)

class LegalQA:
    """Handles question answering using retrieved documents and LLM"""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
    def load_model(self):
        """Load the LLM model and tokenizer"""
        try:
            print(f"Loading model {self.model_name} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Adjust based on available hardware
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            load_in_8bit = self.device == "cuda"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                load_in_8bit=load_in_8bit
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate an answer to the query using the provided context"""
        if not self.pipeline:
            raise ValueError("Model not loaded")
            
        prompt = self._create_prompt(query, context)
        
        try:
            response = self.pipeline(
                prompt,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                return_full_text=False
            )
            return response[0]["generated_text"].strip()
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "I couldn't generate an answer for this question."
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create a prompt for the LLM with the query and context"""
        return f"""<s>[INST] You are a legal assistant analyzing legal documents. 
Answer the question based only on the provided context. Be precise and cite your sources.

Context:
{context}

Question: {query}

Answer: [/INST]"""

class LegalAssistant:
    """Complete legal document QA system with RAG"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.qa_system = LegalQA()
        self.loaded = False
    
    def load_documents(self, file_paths: List[str], persist_dir: str = None):
        """Load and index documents"""
        if not file_paths:
            raise ValueError("No document paths provided")
            
        if persist_dir and os.path.exists(persist_dir):
            try:
                print(f"Loading existing index from {persist_dir}")
                self.vector_store.load_index(persist_dir)
                self.loaded = self.qa_system.load_model()
                print(f"Loaded existing index with {len(self.vector_store.documents)} chunks")
                return
            except Exception as e:
                print(f"Couldn't load existing index: {e}. Creating new index.")
        
        # Process all documents
        all_chunks = []
        for file_path in file_paths:
            print(f"Processing {file_path}")
            documents = self.document_processor.load_document(file_path)
            if documents:
                chunks = self.document_processor.chunk_documents(documents)
                all_chunks.extend(chunks)
        
        if not all_chunks:
            raise ValueError("No valid documents were loaded")
        
        # Build vector index
        print(f"Building index with {len(all_chunks)} chunks")
        self.vector_store.build_index(all_chunks)
        
        # Save index if persist_dir is provided
        if persist_dir:
            self.vector_store.save_index(persist_dir)
            print(f"Saved index to {persist_dir}")
        
        # Load the QA model
        self.loaded = self.qa_system.load_model()
    
    def ask_question(self, question: str, k: int = 3) -> Dict[str, Any]:
        """Answer a legal question based on the loaded documents"""
        if not self.loaded:
            raise ValueError("System not properly loaded")
        
        # Retrieve relevant documents
        search_results = self.vector_store.search(question, k=k)
        
        if not search_results:
            return {
                "answer": "No relevant information found in the documents.",
                "sources": []
            }
        
        # Combine top results into context
        context = "\n\n".join([res["text"] for res in search_results])
        
        # Generate answer
        answer = self.qa_system.generate_answer(question, context)
        
        # Prepare sources with metadata
        sources = [{
            "text": res["text"][:200] + "...",  # Return preview
            "source": res["metadata"]["source"],
            "page": res["metadata"].get("page", "N/A"),
            "score": res["score"]
        } for res in search_results]
        
        return {
            "answer": answer,
            "sources": sources
        }