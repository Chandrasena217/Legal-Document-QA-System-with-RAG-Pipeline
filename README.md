# Legal Document QA System with RAG Pipeline

![Legal Document Analysis](Banner.png) *(placeholder image)*

## Problem Statement

Legal professionals often spend excessive time searching through lengthy documents to find specific clauses, precedents, or answers to legal questions. This system aims to automate that process by:

1. Processing various legal document formats (PDFs, text files)
2. Creating a searchable knowledge base
3. Providing accurate answers to natural language questions
4. Citing relevant sources from the documents

## Solution Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline that:

- Processes and chunks legal documents
- Creates vector embeddings for semantic search
- Uses a large language model (Mistral-7B) to generate precise answers
- Provides source citations for verification

## Key Features

- **Document Processing**: Handles PDFs and text documents with proper text extraction and cleaning
- **Semantic Search**: Finds relevant document chunks using FAISS vector similarity
- **Precise QA**: Generates accurate answers with citations using Mistral-7B
- **API Endpoints**: FastAPI backend for easy integration
- **Persistent Storage**: Saves processed documents for future queries

## Technology Stack

### Core Components
- **Document Processing**: PyMuPDF, LangChain text splitters
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS for efficient similarity search
- **LLM**: Mistral-7B-Instruct for answer generation
- **Backend**: FastAPI with CORS support

### Python Libraries
- Transformers (Hugging Face)
- Sentence Transformers
- FAISS
- LangChain
- FastAPI
![Oroject Structure](Structure.png)
