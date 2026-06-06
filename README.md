# Conversational Hybrid RAG with LangChain

A production-style Conversational Retrieval-Augmented Generation (RAG) application built using LangChain, Qdrant, Hybrid Retrieval (Vector Search + BM25), Gemini, FastAPI, and LangSmith.

The system supports:

* PDF ingestion
* Semantic search using embeddings
* Keyword search using BM25
* Hybrid retrieval
* Conversational memory
* History-aware query rewriting
* FastAPI API endpoint
* LangSmith observability and tracing

You can access the actual pdf on this link - https://rikalp.in/wp-content/uploads/2024/07/HR-Policy.pdf

---

# Architecture

```text
User Question
      │
      ▼
Conversation History
      │
      ▼
History-Aware Retriever
(Query Rewriting)
      │
      ▼
Hybrid Retriever
 ┌──────────────┐
 │ Vector Search│
 │   (Qdrant)   │
 └──────────────┘
         +
 ┌──────────────┐
 │ BM25 Search  │
 └──────────────┘
      │
      ▼
Relevant Context
      │
      ▼
Gemini LLM
      │
      ▼
Final Answer
```

---

# Features

## Conversational RAG

Supports multi-turn conversations.

Example:

```text
User: How many leaves are allowed?

Assistant: 18 earned leaves per year.

User: How many can be carried forward?
```

The assistant understands that the second question refers to leaves from the previous conversation.

---

## History-Aware Retrieval

Uses LangChain's:

```python
create_history_aware_retriever()
```

to rewrite follow-up questions into standalone questions before retrieval.

Example:

```text
How many can be carried forward?
```

becomes:

```text
How many earned leaves can be carried forward to the next year?
```

---

## Hybrid Retrieval

Combines:

### Semantic Search

Powered by:

* HuggingFace Embeddings
* Qdrant Vector Database

Finds semantically similar content.

---

### Keyword Search

Powered by:

* BM25

Finds exact keyword matches.

---

### Ensemble Retriever

Combines both retrieval strategies for better recall and retrieval quality.

---

## FastAPI Integration

The RAG system is exposed through REST APIs.

Example:

```http
POST /chat
```

Request:

```json
{
  "question": "How many leaves are allowed?",
  "session_id": "user_1"
}
```

Response:

```json
{
  "answer": "Employees are entitled to 18 earned leaves per year."
}
```

---

## LangSmith Tracing

Integrated with LangSmith for:

* Prompt inspection
* Query rewriting analysis
* Retriever debugging
* Token tracking
* Latency analysis
* Chain visualization

---

# Tech Stack

## LLM

* Gemini 2.5 Flash

## Framework

* LangChain

## Vector Database

* Qdrant

## Embeddings

* HuggingFace Sentence Transformers

## Retrieval

* BM25
* Ensemble Retriever

## API Layer

* FastAPI

## Observability

* LangSmith

---

# Project Structure

```text
rag-langchain/
│
├── app/
│   ├── api/
│   │   └── main.py
│   │
│   ├── chains/
│   │   └── conversational_rag.py
│   │
│   ├── embeddings/
│   │   └── embedding_model.py
│   │
│   ├── loaders/
│   │   └── pdf_loader.py
│   │
│   ├── memory/
│   │   └── memory_store.py
│   │
│   ├── retrievers/
│   │   └── hybrid_retriever.py
│   │
│   ├── vectordb/
│   │   └── qdrant_client.py
│   │
│   ├── ingest.py
│   │
│   └── main.py
│
├── data/
│
├── qdrant_data/
│
├── requirements.txt
│
├── .env
│
└── README.md
```

---

# Installation

Clone repository:

```bash
git clone <repository-url>
```

Move into project:

```bash
cd rag-langchain
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate virtual environment:

Mac/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_gemini_api_key

LANGSMITH_API_KEY=your_langsmith_api_key

LANGSMITH_TRACING=true

LANGSMITH_PROJECT=rag-langchain
```

---

# Index Documents

Place PDFs inside:

```text
data/
```

Run ingestion:

```bash
python -m app.ingest
```

This will:

* Load PDFs
* Split documents
* Generate embeddings
* Store vectors in Qdrant

---

# Run API

Start FastAPI server:

```bash
uvicorn app.api.main:app
```

API available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Example API Request

```http
POST /chat
```

Request:

```json
{
  "question": "How many leaves are allowed?",
  "session_id": "employee_1"
}
```

Follow-up:

```json
{
  "question": "How many can be carried forward?",
  "session_id": "employee_1"
}
```

The same session ID preserves conversation history.

---

# Future Improvements

* Source citations
* Streaming responses
* Docker deployment
* PostgreSQL chat history
* Reranking layer
* Multi-PDF support
* Authentication
* User management
* Cloud deployment (AWS/GCP/Azure)

---

# Key Learnings

This project demonstrates:

* Retrieval-Augmented Generation (RAG)
* Conversational AI systems
* Hybrid Retrieval techniques
* Vector databases
* LangChain orchestration
* FastAPI integration
* Observability with LangSmith
* Production-oriented AI architecture

```
```
