import logging

from fastapi import APIRouter


logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Health check",
)
async def health() -> dict[str, str]:
    logger.info("Received /health request")
    return {"status": "ok"}