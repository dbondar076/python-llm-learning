# Python LLM Text Analyzer

A production-style learning project that demonstrates how to build an LLM-powered backend service with FastAPI, OpenAI API, structured outputs, evaluation workflows, and Retrieval-Augmented Generation (RAG).


## Features

- LLM text analysis:
  - `/analyze` — returns category + summary
  - `/summarize`
  - `/classify`
  - `/extract-user`
- Batch processing:
  - `/analyze-many`
  - `/analyze-many-safe` — degraded fallback when LLM requests fail
- Robust LLM integration:
  - retries with exponential backoff
  - timeout handling
  - degraded fallback mode
  - dynamic runtime settings support
  - async concurrency control via semaphore
  - TTL caching
  - structured logging
  - request ID tracing
- RAG (Retrieval-Augmented Generation)
  - `/rag/search` — semantic chunk retrieval
  - `/rag/answer` — manual RAG agent
  - `/rag/answer/stream` — streaming responses
  - `/rag/answer/langgraph` — LangGraph-based agent
  - `/rag/answer/langgraph/stream` — LangGraph streaming
- Agent capabilities
  - dynamic routing: `direct` / `clarify` / `rag`
  - conversation-aware question rewriting
  - memory across requests (via LangGraph checkpointer)
  - fallback handling when context is insufficient
- Streaming
  - incremental token streaming via API
  - structured events: `meta`, `chunk`, `done`
- Support for:
  - `top_k`
  - `min_score`
  - `title_filter`
  - `doc_id_filter`
- Chunk merging experiments
- RAG API and dependency override tests


## Agent Architecture

The project includes two RAG agent implementations:

### 1. Manual Agent
Custom orchestration with explicit control over:
- routing
- memory handling
- retrieval and answer generation

### 2. LangGraph Agent
A production-style graph-based agent using LangGraph:
- nodes (route, retrieve, answer, fallback)
- conditional edges
- state management
- built-in checkpoint-based memory

### Flow
User question
- memory-aware rewrite (if needed)
- route decision (LLM + heuristics)
  - direct → answer
  - clarify → clarification
  - rag → retrieval → answer / fallback

### Memory

- conversation state stored via LangGraph checkpointer (thread_id-based sessions)
- messages tracked using:
  - `HumanMessage`
  - `AIMessage`
- previous context used for:
  - resolving ambiguous queries
  - improving retrieval quality
  - disambiguating follow-up questions ("what about that?")

### Streaming

LangGraph agent supports streaming responses:

- transport: NDJSON (newline-delimited JSON)
- implemented via `graph.astream(...)`


## Prompt Engineering & Evaluation

- Prompt versions (`v1`, `v2`)
- Dataset-based evaluation (`eval_cases.json`)
- Metrics:
  - pass rate
  - avg words / chars
  - latency
- LLM-as-judge for prompt comparison
- Prompt tournament experiments


## Project Structure

```
app/
  api.py
  models.py
  dependencies.py
  error_handlers.py
  request_context.py
  agents/
    rag/
      state.py
      nodes.py
      edges.py
      graph.py
      runtime.py
      response.py
  routers/
    analysis.py
    rag.py
    health.py
  services/
    agent_runtime.py
    analyzer.py
    conversation_memory.py
    llm_service.py
    llm_prompts.py
    llm_parsers.py
    llm_cache.py
    llm_errors.py
    llm_schemas.py
    logging_config.py
    manual_agent_service.py
    openai_client.py
    prompt_registry.py
    rag_answer_service.py
    rag_index_service.py
    rag_retrieval_service.py
    rag_tools.py
tests/
  test_api.py
  test_rag_api.py
  test_rag_api_fast.py
  test_rag_api_dependency_override.py
  test_rag_search_api.py
  test_retry_timeout.py
  test_backoff_degraded.py
  test_eval_summary.py

experiments/
  lesson*.py
```

## Setup

### 1. Clone repo

```bash
git clone <your-repo> 
cd python-llm-learning 
```
### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```
### 3. Install dependencies

```bash
pip install -r requirements.txt 
```
### 4. Create .env

```env
OPENAI_API_KEY=your_api_key_here 
```
### Add other settings if needed, for example:

```env
USE_REAL_LLM=true
LLM_TIMEOUT_SECONDS=30
LLM_MAX_RETRIES=3
LLM_CONCURRENCY_LIMIT=5
```

## Run the API

```bash
uvicorn app.api:app --reload 
```
### Swagger UI:
```text
 http://127.0.0.1:8000/docs 
```

## Run Tests

### Fast tests

```bash
pytest -m fast 
```
### Integration tests

```bash
pytest -m integration 
```
### All tests

```bash
pytest 
```
### Optional future marker

If real OpenAI smoke tests are added:
```bash
pytest -m real_llm
```

## Run Experiments

### Prompt evaluation

```bash 
python experiments/lesson46.py 
```

### Prompt Tournament (LLM-as-judge)

```bash
python experiments/lesson48.py 
```


## Example API Response

```json
{
  "text": "What is Python?",
  "category": "question",
  "summary": "Asking what Python is"
}
```

## Example RAG Response
```json
{
  "answer": "Python can be used for web development, automation, data analysis, and AI.",
  "chunks": [
    {
      "doc_id": "doc1",
      "title": "Python",
      "chunk_id": "doc1_chunk_1",
      "text": "Python is a high-level programming language.",
      "score": 0.55
    }
  ]
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

## Key Concepts Covered

- Prompt engineering
- Structured LLM output (JSON schema)
- Retry / timeout handling
- Degraded fallback patterns
- Observability (logging + request ID)
- Evaluation pipelines
- LLM-as-judge
- Prompt experimentation
- Embeddings
- Semantic search
- Retrieval-Augmented Generation (RAG)
- Agent-based architectures
- Graph-based orchestration (LangGraph)
- Conversational memory
- Streaming LLM responses

## Manual vs LangGraph Agent

| Feature         | Manual Agent | LangGraph Agent |
|-----------------|-------------|-----------------|
| Control         | High        | Medium          |
| Abstraction     | Low         | High            |
| Memory          | Custom      | Built-in        |
| Streaming       | Manual      | Native          |
| Scalability     | Medium      | High            |
| Debuggability   | High        | Medium          |

## Roadmap

- Real LLM smoke tests
- Better dataset coverage for evaluation
- Improved chunking strategies
- Optional vector database integration

## Why This Project

This project demonstrates how to evolve from a simple LLM integration to a production-style agent system:

- start with prompt-based APIs
- add reliability (retry, timeout, caching)
- introduce evaluation workflows
- build RAG pipelines
- evolve into agent-based architectures
- migrate to graph-based orchestration (LangGraph)

It is designed as a learning path for backend engineers entering the LLM space.

## License

MIT