import logging

from fastapi import APIRouter, Depends

from app.agents.tools_demo.runtime import run_tools_demo_agent
from app.dependencies import get_rag_records
from app.models import ToolsDemoRequest, ToolsDemoResponse
from app.services.rag_index_service import ChunkEmbeddingRecord

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Tools Demo"])


@router.post(
    "/tools-demo/answer",
    response_model=ToolsDemoResponse,
    summary="Answer using the tools demo agent",
    description="Route a question to calculator, list_docs, search_chunks, or direct answer.",
)
async def tools_demo_answer(
    request: ToolsDemoRequest,
    records: list[ChunkEmbeddingRecord] = Depends(get_rag_records),
) -> ToolsDemoResponse:
    logger.info("Received /tools-demo/answer request: %s", request.question)

    result = await run_tools_demo_agent(
        question=request.question,
        records=records,
        top_k=request.top_k,
    )

    return ToolsDemoResponse(
        answer=result.get("answer", ""),
        route=result.get("route"),
        selected_tool=result.get("selected_tool"),
        tool_input=result.get("tool_input"),
        tool_output=result.get("tool_output"),
    )