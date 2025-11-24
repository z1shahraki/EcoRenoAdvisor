# EcoRenoAdvisor

A local RAG + agent system for personalized, eco-friendly renovation decisions. Combines materials filtering, document retrieval, and local LLM to provide document-grounded renovation advice.

## What This Project Demonstrates

- **Local LLM Integration**: Direct llama_cpp usage (Qwen2.5 3B) with fallback server mode
- **RAG Pipeline**: BGE embeddings (sentence-transformers) + in-memory vector search (Qdrant optional)
- **Agent Pattern**: Combines structured filtering (pandas) with semantic search
- **Testing**: Pytest-based unit and integration tests
- **Clean Architecture**: WSL-compatible, uses uv package manager

## Quick Start

### Step 1: Setup with uv

```bash
cd EcoRenoAdvisor
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Step 2: Download Model

Download `qwen2.5-3b-instruct-q4_k_m.gguf` (2GB) to `models/` directory:
- https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF

### Step 3: Run UI

```bash
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
.venv/bin/python ui/app.py
```

Open browser: **http://localhost:7860**

**Note:** Uses direct mode by default - model loads automatically when first used. No separate server needed.

## Testing

### Unit Tests (Fast, No Services)

```bash
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
.venv/bin/pytest tests/ -v -m "not integration"
```

**Tests:** Materials filtering, chunking, embeddings, agent wiring, UI construction

### Integration Test (RAG + LLM End-to-End)

```bash
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
.venv/bin/pytest tests/test_llm_simple_rag.py -v -s
```

**Requires:** Model file in `models/qwen2.5-3b-instruct-q4_k_m.gguf`

This test proves LLM + RAG works end-to-end: embeddings → retrieval → LLM generation.

### Demo Script

```bash
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
.venv/bin/python demo_rag.py
```

Shows RAG + LLM flow with human-readable output.

## Architecture

```
data/raw/          → CSV, PDFs
    ↓
ingestion/         → Clean, extract, chunk
    ↓
data/clean/        → Parquet, JSONL
    ↓
rag/               → Embeddings → In-memory vector search (Qdrant optional)
    ↓
agent/             → Filter + RAG + LLM
    ↓
ui/                → Gradio interface
```

## How It Works

1. **Materials Filtering**: Filters renovation materials by category, price, eco-score, and VOC level using pandas
2. **Document Retrieval**: Uses BGE embeddings to find relevant document chunks via semantic search
3. **LLM Generation**: Combines filtered materials + retrieved documents + user query → generates personalized recommendations
4. **Vector Search**: Works with in-memory search (no Docker needed) or Qdrant for production

## Modes

### Direct Mode (Default)

Model loaded directly via llama_cpp - no server needed. Perfect for portfolio demos.

```python
from agent.agent import RenovationAgent
agent = RenovationAgent(mode="direct")
```

### Server Mode (Optional)

For production/multi-user scenarios:

```bash
.venv/bin/python -m llama_cpp.server --model models/qwen2.5-3b-instruct-q4_k_m.gguf --port 8000
```

```python
agent = RenovationAgent(mode="server")
```

## Requirements

- Python 3.10+
- uv package manager
- Docker (optional, for Qdrant vector DB)
- LLM model file in `models/` directory

## License

Open source - portfolio use.
