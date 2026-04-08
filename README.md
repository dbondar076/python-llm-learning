# LLM RAG Backend (FastAPI + LangGraph)

Production-style backend service demonstrating how to build reliable LLM-powered systems with Retrieval-Augmented Generation (RAG), agent routing, evaluation workflows, and streaming.

## Overview

This project explores how to evolve a simple LLM integration into a robust, production-ready backend system.

It covers the full lifecycle:
- prompt-based APIs →
- reliability layer (retry, timeout, caching) →
- evaluation pipelines →
- RAG →
- agent-based orchestration →
- graph-based execution (LangGraph)

The goal is not just to call an LLM, but to control, evaluate, and scale its behavior.

## Key Capabilities

### LLM APIs
- /analyze, /summarize, /classify
- structured JSON outputs (schema-validated)

### Reliability Layer
- retries with exponential backoff
- timeout handling
- degraded fallback mode
- async concurrency control (semaphore)
- TTL caching
- request tracing (request_id)
- structured logging

### RAG (Retrieval-Augmented Generation)
- semantic chunk retrieval
- configurable filters (top_k, min_score, etc.)
- chunk merging strategies
- answer generation with context grounding

### Agent System
- dynamic routing:
- direct — answer without retrieval
- clarify — ask follow-up question
- rag — retrieve + answer
- conversation-aware query rewriting
- fallback when context is insufficient

### LangGraph Integration
- graph-based orchestration
- explicit nodes: route → retrieve → answer → fallback
- conditional edges
- checkpoint-based memory (thread sessions)

### Streaming
- token streaming via NDJSON
- structured events: meta, chunk, done

## Architecture
The system is built around a multi-stage LLM pipeline:
### 1.	Input
- user query
- optional conversation history
### Rewrite (optional)
- resolves ambiguity using memory
### Routing
- LLM + heuristics decide:
- direct / clarify / rag
### Retrieval (RAG path)
- semantic search over chunks
- filtering + ranking
### Answer Generation
- grounded in retrieved context
- fallback if insufficient
### Streaming (optional)
- incremental token output

## Agent Implementations
### Manual Agent
- explicit orchestration
- full control over flow
- easier to debug

### LangGraph Agent
- declarative graph execution
- built-in state management
- persistent memory via checkpointer

| Feature         | Manual Agent | LangGraph Agent |
|-----------------|-------------|-----------------|
| Control         | High        | Medium          |
| Abstraction     | Low         | High            |
| Memory          | Custom      | Built-in        |
| Streaming       | Manual      | Native          |
| Scalability     | Medium      | High            |
| Debuggability   | High        | Medium          |

### Evaluation & Prompt Engineering
- versioned prompts (v1, v2)
- dataset-driven evaluation
- metrics:
  - pass rate
  - latency
  - response size
- LLM-as-judge
- prompt tournament experiments

This enables systematic prompt improvement, not trial-and-error.


## Project Structure

```
app/
  api.py
  routers/
  services/
    llm_service.py
    rag_retrieval_service.py
    rag_answer_service.py
    manual_agent_service.py
  agents/
    rag/
      graph.py
      nodes.py
      edges.py

tests/
experiments/
```

## Getting Started

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### Configure

```env
OPENAI_API_KEY=your_api_key
USE_REAL_LLM=true
LLM_TIMEOUT_SECONDS=30
LLM_MAX_RETRIES=3
LLM_CONCURRENCY_LIMIT=5
```
### Run

```bash
uvicorn app.api:app --reload
```
### Swagger:

```code
http://127.0.0.1:8000/docs
```
### Testing

```bash
pytest              # all tests
pytest -m fast      # fast unit tests
pytest -m integration
```

## Example

### RAG Answer

```json
{
  "answer": "Python can be used for web development, automation, data analysis, and AI.",
  "chunks": [...]
}
```

## Tech Stack

- Python 3.12
- FastAPI
- OpenAI API
- asyncio
- pytest
- LangGraph
- LangChain Core (messages abstraction)
- pytest

## Design Focus

This project focuses on engineering challenges of LLM systems:

- controlling non-deterministic outputs
- handling failures and timeouts
- grounding answers in data (RAG)
- routing between multiple strategies
- evaluating prompt quality
- managing conversational context

## Roadmap

- vector DB integration
- improved retrieval scoring
- stronger evaluation datasets
- real LLM smoke tests

## Why This Project Matters

Most LLM demos stop at “call the model”.

This project explores what comes next:
- reliability
- evaluation
- architecture
- scalability

It is designed as a bridge between simple LLM usage and production systems.

## License

MIT