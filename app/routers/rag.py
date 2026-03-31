import json
import logging

from collections.abc import AsyncGenerator
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.agents.rag.response import build_langgraph_response
from app.agents.rag.runtime import run_langgraph_agent, prepare_langgraph_stream
from app.dependencies import get_rag_records, get_retriever
from app.models import RagAnswerRequest, RagChunk, RagResponse, RagSearchRequest, RagSearchResponse
from app.services.manual_agent_service import run_rag_agent
from app.services.rag_index_service import ChunkEmbeddingRecord
from app.services.rag_retrieval_service import retrieve_top_chunks


logger = logging.getLogger(__name__)

router = APIRouter(tags=["RAG"])


def build_rag_chunks(chunks: list[dict]) -> list[RagChunk]:
    return [
        RagChunk(
            doc_id=c["doc_id"],
            title=c["title"],
            chunk_id=c["chunk_id"],
            text=c["text"],
            score=c["score"],
        )
        for c in chunks
    ]


def serialize_chunks(chunks: list[dict]) -> list[dict]:
    return [
        {
            "doc_id": c["doc_id"],
            "title": c["title"],
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "score": c["score"],
        }
        for c in chunks
    ]


async def stream_text_chunks(
    text: str,
    chunk_size: int = 20,
) -> AsyncGenerator[str, None]:
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        yield json.dumps(
            {
                "type": "chunk",
                "content": chunk,
            }
        ) + "\n"


@router.post(
    "/rag/search",
    response_model=RagSearchResponse,
    summary="Search relevant chunks",
    description="Retrieve the most relevant chunks for a question using semantic search.",
)
async def rag_search(
    request: RagSearchRequest,
    retriever = Depends(get_retriever),
) -> RagSearchResponse:
    logger.info("Received /rag/search request: %s", request.question)

    chunks = retriever.search(
        query=request.question,
        top_k=request.top_k,
        title_filter=request.title_filter,
        doc_id_filter=request.doc_id_filter,
    )

    response_chunks = build_rag_chunks(chunks)
    return RagSearchResponse(chunks=response_chunks)


@router.post(
    "/rag/answer",
    response_model=RagResponse,
    summary="Answer question with RAG",
    description="Retrieve relevant chunks and generate a grounded answer based on the provided context.",
)
async def rag_answer(
    request: RagAnswerRequest,
    records: list[ChunkEmbeddingRecord] = Depends(get_rag_records),
) -> RagResponse:
    logger.info("Received /rag/answer request: %s", request.question)

    chunks, answer, meta = await run_rag_agent(
        question=request.question,
        records=records,
        session_id=request.session_id,
        top_k=request.top_k,
        min_score=request.min_score,
        title_filter=request.title_filter,
        doc_id_filter=request.doc_id_filter,
    )

    response_chunks = build_rag_chunks(chunks)

    return RagResponse(
        answer=answer,
        chunks=response_chunks,
        meta=meta,
    )


@router.post(
    "/rag/answer/langgraph",
    response_model=RagResponse,
    summary="Answer question with RAG via LangGraph",
    description="Retrieve relevant chunks and generate a grounded answer using the LangGraph agent.",
)
async def rag_answer_langgraph(
    request: RagAnswerRequest,
    records: list[ChunkEmbeddingRecord] = Depends(get_rag_records),
) -> RagResponse:
    logger.info("Received /rag/answer/langgraph request: %s", request.question)

    state = await run_langgraph_agent(
        question=request.question,
        records=records,
        session_id=request.session_id,
        top_k=request.top_k,
        min_score=request.min_score,
        title_filter=request.title_filter,
        doc_id_filter=request.doc_id_filter,
    )

    result = build_langgraph_response(state)

    response_chunks = build_rag_chunks(result["chunks"])

    return RagResponse(
        answer=result["answer"],
        chunks=response_chunks,
        meta=result["meta"],
    )


@router.post(
    "/rag/answer/stream",
    summary="Answer question with RAG and stream the result",
    description="Retrieve relevant chunks and stream the grounded answer in chunks.",
)
async def rag_answer_stream(
    request: RagAnswerRequest,
    records: list[ChunkEmbeddingRecord] = Depends(get_rag_records),
) -> StreamingResponse:
    logger.info("Received /rag/answer/stream request: %s", request.question)

    chunks, answer, meta = await run_rag_agent(
        question=request.question,
        records=records,
        session_id=request.session_id,
        top_k=request.top_k,
        min_score=request.min_score,
        title_filter=request.title_filter,
        doc_id_filter=request.doc_id_filter,
    )

    async def event_stream():
        yield json.dumps(
            {
                "type": "meta",
                "meta": meta,
                "chunks": serialize_chunks(chunks),
            }
        ) + "\n"

        async for item in stream_text_chunks(answer):
            yield item

        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
    )


@router.post(
    "/rag/answer/langgraph/stream",
    summary="Stream answer with RAG via LangGraph",
    description="Retrieve relevant chunks and stream a grounded answer using the LangGraph agent.",
)
async def rag_answer_langgraph_stream(
    request: RagAnswerRequest,
    records: list[ChunkEmbeddingRecord] = Depends(get_rag_records),
) -> StreamingResponse:
    logger.info("Received /rag/answer/langgraph/stream request: %s", request.question)

    async def event_stream():
        graph, config, initial_state = await prepare_langgraph_stream(
            question=request.question,
            records=records,
            session_id=request.session_id,
            top_k=request.top_k,
            min_score=request.min_score,
            title_filter=request.title_filter,
            doc_id_filter=request.doc_id_filter,
        )

        async for _ in graph.astream(
            initial_state,
            config=config,
            stream_mode="updates",
        ):
            pass

        snapshot = await graph.aget_state(config)
        state = snapshot.values if snapshot and snapshot.values else {}
        result = build_langgraph_response(state)

        yield json.dumps(
            {
                "type": "meta",
                "meta": result["meta"],
                "chunks": result["chunks"],
            }
        ) + "\n"

        async for item in stream_text_chunks(result["answer"]):
            yield item

        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
    )