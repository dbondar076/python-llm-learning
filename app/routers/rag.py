import asyncio
import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.dependencies import get_rag_records
from app.models import RagAnswerRequest, RagChunk, RagResponse, RagSearchRequest, RagSearchResponse
from app.services.agent_service import run_rag_agent
# from app.services.langgraph_rag_agent import run_rag_agent_langgraph
from app.services.rag_index_service import ChunkEmbeddingRecord
from app.services.rag_retrieval_service import retrieve_top_chunks


logger = logging.getLogger(__name__)

router = APIRouter(tags=["RAG"])


async def stream_text_chunks(answer: str, chunk_size: int = 20):
    for i in range(0, len(answer), chunk_size):
        chunk = answer[i:i + chunk_size]

        yield json.dumps(
            {
                "type": "chunk",
                "content": chunk,
            }
        ) + "\n"

        await asyncio.sleep(0.05)

    yield json.dumps(
        {
            "type": "done",
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
    records: list[ChunkEmbeddingRecord] = Depends(get_rag_records),
) -> RagSearchResponse:
    logger.info("Received /rag/search request: %s", request.question)

    chunks = retrieve_top_chunks(
        query=request.question,
        records=records,
        top_k=request.top_k,
        title_filter=request.title_filter,
        doc_id_filter=request.doc_id_filter,
    )

    response_chunks = [
        RagChunk(
            doc_id=c["doc_id"],
            title=c["title"],
            chunk_id=c["chunk_id"],
            text=c["text"],
            score=c["score"],
        )
        for c in chunks
    ]

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

    chunks, answer = await run_rag_agent(
        question=request.question,
        records=records,
        top_k=request.top_k,
        min_score=request.min_score,
        title_filter=request.title_filter,
        doc_id_filter=request.doc_id_filter,
    )

    # chunks, answer = await run_rag_agent_langgraph(
    #     question=request.question,
    #     records=records,
    #     top_k=request.top_k,
    #     min_score=request.min_score,
    #     title_filter=request.title_filter,
    #     doc_id_filter=request.doc_id_filter,
    # )

    response_chunks = [
        RagChunk(
            doc_id=c["doc_id"],
            title=c["title"],
            chunk_id=c["chunk_id"],
            text=c["text"],
            score=c["score"],
        )
        for c in chunks
    ]

    return RagResponse(
        answer=answer,
        chunks=response_chunks,
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

    chunks, answer = await run_rag_agent( #run_rag_agent_langgraph(
        question=request.question,
        records=records,
        top_k=request.top_k,
        min_score=request.min_score,
        title_filter=request.title_filter,
        doc_id_filter=request.doc_id_filter,
    )

    async def event_stream():
        yield json.dumps(
            {
                "type": "meta",
                "chunks": [
                    {
                        "doc_id": c["doc_id"],
                        "title": c["title"],
                        "chunk_id": c["chunk_id"],
                        "text": c["text"],
                        "score": c["score"],
                    }
                    for c in chunks
                ],
            }
        ) + "\n"

        async for item in stream_text_chunks(answer):
            yield item

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
    )