# Gene Editing Almanac (RAG System)

A next-generation Retrieval-Augmented Generation (RAG) system specialized in Gene Editing (CRISPR, Base Editing, Prime Editing) and Clinical Data.

## Features

- **Hybrid Retrieval**: Combines semantic search (FAISS) with metadata filtering.
- **Dual Knowledge Base**: 
  - **Main Database**: Pre-indexed high-quality literature.
  - **User Library**: Upload your own RIS/BibTeX files for private, session-specific analysis.
- **Intelligent Routing**: LLM-driven intent analysis to switch between general knowledge queries and specific user-upload analysis.
- **Reference Provenance**: Strict citation tracking and preventing hallucinations by anchoring answers to retrieved chunks.
- **Modern UI**: React + TypeScript frontend with persistent chat history.

## Tech Stack

- **Backend**: Python, FastAPI, LangChain, FAISS
- **Frontend**: React, TypeScript, Vite, Tailwind CSS
- **LLM**: Compatible with OpenAI-compatible APIs (DeepSeek, GPT-4, etc.)

## Setup

1. **Backend Setup**:
   ```bash
   pip install -r requirements.txt
   # Set your API keys in .env
   python app.py
   ```

2. **Frontend Setup**:
   ```bash
   cd frontend
   npm install
   npm run build
   ```

3. **Access**:
   Open `http://localhost:6006` in your browser.

## Directory Structure

- `core/`: RAG logic (retrieval, decision engine, risk assessment).
- `data/`: Vector databases and PDF storage.
- `frontend/`: React application source code.
- `scripts/`: Data ingestion scripts.
