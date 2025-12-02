# Bootstrap_GenAI

A Retrieval-Augmented Generation (RAG) application built with FastAPI, LangChain, and OpenAI.

## Overview

This project implements a RAG system that combines document retrieval with generative AI to answer questions based on custom knowledge bases. The API uses FAISS for vector similarity search and GPT-3.5-turbo for generating contextually relevant answers.

## Project Structure

```
Bootstrap_GenAI/
├── README.md
├── .env                          # Environment variables (not included in repo)
├── .venv/                        # Virtual environment
└── Assignment/
    ├── rag_app.py               # Main FastAPI application
    └── knowledge_base.txt       # Document knowledge base (auto-generated)
```

## Features

- **FastAPI REST API** - Simple endpoint for querying the knowledge base
- **LangChain Integration** - Streamlined chain for RAG pipeline
- **Vector Embeddings** - OpenAI embeddings for semantic search
- **FAISS Vector Store** - Efficient similarity search
- **Auto-generated Knowledge Base** - Default Apollo 11 sample data

## Requirements

- Python 3.14+
- FastAPI
- Uvicorn
- LangChain & LangChain community
- LangChain OpenAI integration
- LangChain text splitters
- FAISS (CPU version)
- python-dotenv
- OpenAI API key

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Bootstrap_GenAI
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # Windows PowerShell
   # or
   source .venv/bin/activate     # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install fastapi uvicorn langchain langchain-openai langchain-community langchain-text-splitters faiss-cpu python-dotenv
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

1. **Navigate to Assignment directory**
   ```bash
   cd Assignment
   ```

2. **Start the server**
   ```bash
   uvicorn rag_app:app --reload
   ```

3. **Access the API**
   - API runs on `http://127.0.0.1:8000`
   - Swagger UI: `http://127.0.0.1:8000/docs`
   - ReDoc: `http://127.0.0.1:8000/redoc`

## API Endpoints

### POST `/ask`
Ask a question to the RAG system.

**Request:**
```json
{
  "question": "Who landed on the Moon?"
}
```

**Response:**
```json
{
  "answer": "Neil Armstrong and Buzz Aldrin landed on the Moon during the Apollo 11 mission on July 20, 1969."
}
```

## How It Works

1. **Knowledge Base Initialization** - Loads and splits documents into chunks
2. **Embedding Generation** - Converts text chunks to vector embeddings using OpenAI
3. **Vector Store** - Stores embeddings in FAISS for fast retrieval
4. **Query Processing** - When a question is asked:
   - The question is embedded
   - Similar documents are retrieved from FAISS
   - Retrieved context is passed to GPT-3.5-turbo
   - The model generates an answer based on the context

## Customization

To use your own knowledge base:

1. Replace `knowledge_base.txt` with your own document
2. Modify the `DOC_PATH` variable in `rag_app.py` if needed
3. Restart the server

## Notes

- The application uses GPT-3.5-turbo model (configurable in the code)
- FAISS CPU version is used; for GPU acceleration, install `faiss-gpu`
- Python 3.14 compatibility warning about Pydantic V1 can be safely ignored
- Answers are strictly based on the context provided in the knowledge base

## Troubleshooting

**Virtual environment not activated:**
- Windows: `.\.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

**uvicorn command not found:**
- Ensure virtual environment is activated
- Reinstall uvicorn: `pip install uvicorn`

**OPENAI_API_KEY not found:**
- Create `.env` file in root directory with your API key
- Verify the `.env` file is in the correct location

**Import errors:**
- Install all dependencies: `pip install -r requirements.txt` (if created)
- Or manually install packages from the Requirements section