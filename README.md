# Advanced Hybrid RAG

A hybrid Retrieval-Augmented Generation (RAG) system combining dense vector search (ChromaDB) and sparse keyword search (BM25) via an ensemble retriever. Built for efficient, accurate document querying from a local knowledge base.

## How It Works

```
PDF Folder → Load → Chunk → Embed → ChromaDB (dense)
                  → Load →         BM25 (sparse)
                                      ↓
User Query → Ensemble Retriever (0.5 dense + 0.5 sparse)
                                      ↓
                           Top-K Relevant Chunks
                                      ↓
                        LLM (llama3.2 via Ollama)
                                      ↓
                               Grounded Answer
```

## Features

- **Hybrid Retrieval** — BM25 for keyword matching + ChromaDB for semantic search, combined via ensemble
- **Persistent Vector Store** — ChromaDB persists to disk, no re-embedding on restart
- **Local Knowledge Base** — drop PDFs into a folder, auto-ingested on first run
- **Fully Local** — no API keys needed for inference, runs via Ollama
- **LangSmith Tracing** — optional observability via `@traceable`

## Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama (llama3.2) |
| Embeddings | Ollama (nomic-embed-text) |
| Dense Retriever | ChromaDB |
| Sparse Retriever | BM25 (rank-bm25) |
| Framework | LangChain |
| Observability | LangSmith |

## Getting Started

### Prerequisites

- Python 3.11+
- Ollama with models pulled

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### Setup

1. Clone the repo

```bash
git clone https://github.com/Albert331/Advanced_hybrid_RAG.git
cd Advanced_hybrid_RAG
```

2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Create `.env` file (optional, for LangSmith tracing)

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=hybrid-rag
```

5. Add PDFs to `Knowledge_base/` folder

6. Run

```bash
python rag_tests.py
```

## Usage

```python
from rag_tests import RAG, llm_call

# First run: ingests and embeds all PDFs
# Subsequent runs: loads from ChromaDB cache

llm_call("what optimizer did the authors use?")
# → The Adam optimizer was used, with β1=0.9, β2=0.98, ε=10⁻⁹
```

## Project Structure

```
Advanced_hybrid_RAG/
├── Knowledge_base/      # Drop PDFs here
├── chroma_db/           # Auto-generated, ChromaDB persistence
├── rag_tests.py         # RAG class + llm_call
├── .gitignore
├── .env                 # Not committed
└── requirements.txt
```

## Retrieval Strategy

The ensemble retriever combines two complementary approaches:

- **BM25 (sparse)** — exact keyword matching, great for specific terms, names, numbers
- **ChromaDB (dense)** — semantic similarity, great for conceptual queries

Both are weighted equally (0.5/0.5) with deduplication to avoid repeated chunks.
