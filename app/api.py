import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.error_handlers import register_exception_handlers
from app.request_context import request_id_var
from app.routers.analysis import router as analysis_router
from app.routers.health import router as health_router
from app.routers.rag import router as rag_router
from app.routers import tools_demo, tools_loop_demo
from app.services.rag_index_service import load_chunk_embeddings
from app.services.retrievers.factory import build_retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading RAG index...")
    records = load_chunk_embeddings()
    app.state.rag_records = records
    app.state.retriever = build_retriever(records)
    logger.info("RAG index loaded")

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="Python LLM Learning API",
    description="API for text analysis, summarization, classification, extraction, and RAG search/answer experiments.",
    version="0.1.0",
    lifespan=lifespan,
)

register_exception_handlers(app)


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    token = request_id_var.set(request_id)

    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        request_id_var.reset(token)


app.include_router(health_router)
app.include_router(analysis_router)
app.include_router(rag_router)
app.include_router(tools_demo.router)
app.include_router(tools_loop_demo.router)