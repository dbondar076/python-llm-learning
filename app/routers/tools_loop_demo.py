import logging

from fastapi import APIRouter, Depends

from app.dependencies import get_rag_records
from app.models import ToolsLoopDemoRequest, ToolsLoopDemoResponse
from app.agents.tools_loop_demo.runtime import run_tools_loop_demo_agent
from app.services.rag_index_service import ChunkEmbeddingRecord


logger = logging.getLogger(__name__)

router = APIRouter(tags=["Tools Loop Demo"])


@router.post(
    "/tools-loop-demo/answer",
    response_model=ToolsLoopDemoResponse,
    summary="Run tools loop demo agent",
)
async def tools_loop_demo_answer(
    request: ToolsLoopDemoRequest,
    records: list[ChunkEmbeddingRecord] = Depends(get_rag_records),
) -> ToolsLoopDemoResponse:
    logger.info("Received /tools-loop-demo/answer: %s", request.question)

    result = await run_tools_loop_demo_agent(
        question=request.question,
        records=records,
        top_k=request.top_k,
        max_steps=request.max_steps,
    )

    return ToolsLoopDemoResponse(**result)