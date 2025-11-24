# EcoRenoAdvisor

EcoRenoAdvisor is a local renovation advisor that combines document grounded reasoning, materials filtering, and a lightweight local LLM. It answers renovation questions with evidence taken from your documents and material database. The goal is to test feasibility of a private, offline assistant that supports sustainable renovation planning.

## Why This Project Exists

Many homeowners want renovation guidance that is personalised, environmentally aware, and grounded in real information rather than generic suggestions. Most tools online are commercial, limited, or require cloud access. EcoRenoAdvisor explores whether a small local LLM, combined with a simple RAG pipeline, can:

- read and understand renovation documents
- filter materials by sustainability indicators
- combine both to produce safe and useful recommendations

This project is a first step toward a private home renovation assistant that runs locally and can be improved over time.

## What the System Demonstrates

- Local LLM integration using llama_cpp and Qwen2.5 3B
- Semantic search over your documents using BGE embeddings
- A basic RAG pipeline that does not depend on any external service
- Material filtering using price, eco score, and VOC levels
- A small agent that blends structured filters with retrieval
- A clean, testable architecture that works in WSL

This version focuses on feasibility and clarity rather than performance, which can be improved later.

## Quick Start

### Setup with uv

```bash
cd EcoRenoAdvisor
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Download Model

Place this file in `models/`:

`qwen2.5-3b-instruct-q4_k_m.gguf`

Download link:
https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF

### Run the UI

```bash
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
.venv/bin/python ui/app.py
```

Open:
**http://localhost:7860**

The model loads automatically when needed. No server required.

## Testing

### Unit Tests

```bash
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
.venv/bin/pytest tests/ -v -m "not integration"
```

Covers filtering, embeddings, chunking, and agent logic.

### Integration Test

```bash
.venv/bin/pytest tests/test_llm_simple_rag.py -v -s
```

Runs full RAG flow with the local model.

### Demo Script

```bash
.venv/bin/python demo_rag.py
```

Shows RAG output in the terminal.

## Architecture Overview

```
data/raw          → PDFs, CSVs
ingestion         → cleaning and chunking
data/clean        → parquet and JSONL
rag               → embeddings and vector search
agent             → filter + retrieve + LLM reasoning
ui                → Gradio interface
```

## How It Works

1. Filters materials using price, category, eco score, VOC level
2. Retrieves relevant document chunks using BGE embeddings
3. Feeds both into a local Qwen model
4. Produces grounded renovation advice

Works with in-memory search or Qdrant if needed.

## Modes

### Direct mode

```python
from agent.agent import RenovationAgent
agent = RenovationAgent(mode="direct")
```

### Server mode

```bash
python -m llama_cpp.server --model models/qwen2.5-3b-instruct-q4_k_m.gguf --port 8000
```

```python
agent = RenovationAgent(mode="server")
```

## Future Enhancements

The current version focuses on feasibility. Next steps that make sense:

- improve retrieval quality
- add multi document context windows
- introduce richer material scoring
- run more efficient models
- add caching for repeated queries

## Requirements

- Python 3.10+
- uv
- LLM model file in `models/`

## License

Open source for portfolio demonstration.
