import os

from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  # app/
DATA_DIR = BASE_DIR / "data"

RAG_INDEX_FILE = os.getenv("RAG_INDEX_FILE", "chunk_embeddings.json")
RAG_INDEX_VERSION = os.getenv("RAG_INDEX_VERSION", "v1")

CHUNK_EMBEDDINGS_PATH = (
    DATA_DIR / "rag" / RAG_INDEX_VERSION / RAG_INDEX_FILE
)

if not CHUNK_EMBEDDINGS_PATH.exists():
    raise FileNotFoundError(
        f"Embeddings file not found: {CHUNK_EMBEDDINGS_PATH}"
    )

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_REAL_LLM = os.getenv("USE_REAL_LLM", "true").lower() == "true"

LLM_CONCURRENCY_LIMIT = int(os.getenv("LLM_CONCURRENCY_LIMIT", "3"))
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_BASE_DELAY_SECONDS = float(os.getenv("LLM_BASE_DELAY_SECONDS", "0.5"))

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))

SUMMARY_PROMPT_VERSION = os.getenv("SUMMARY_PROMPT_VERSION", "v2")
CLASSIFICATION_PROMPT_VERSION = os.getenv("CLASSIFICATION_PROMPT_VERSION", "v1")
ANALYSIS_PROMPT_VERSION = os.getenv("ANALYSIS_PROMPT_VERSION", "v1")
EXTRACTION_PROMPT_VERSION = os.getenv("EXTRACTION_PROMPT_VERSION", "v1")

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.52"))

VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "local")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_chunks")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", ".chroma")