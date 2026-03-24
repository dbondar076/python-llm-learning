import logging
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from app.models import SafeTextAnalysis, TextAnalysis, UserExtractResult
from app.services.analyzer import (
    analyze_many_async,
    analyze_many_safe_async,
    analyze_text_async,
    classify_text_async,
    extract_user_async,
    summarize_text_async,
)


logger = logging.getLogger(__name__)

router = APIRouter(tags=["Analysis"])


class TextRequest(BaseModel):
    text: str


class SummaryResponse(BaseModel):
    summary: str


class ClassificationResponse(BaseModel):
    category: Literal["question", "statement", "short"]


class AnalyzeManyRequest(BaseModel):
    texts: list[str]


class AnalyzeManyResponse(BaseModel):
    results: list[TextAnalysis]


class AnalyzeManySafeResponse(BaseModel):
    results: list[SafeTextAnalysis]


@router.post(
    "/analyze",
    response_model=TextAnalysis,
    summary="Analyze text",
    description="Analyze a text and return category plus summary.",
)
async def analyze(request: TextRequest) -> TextAnalysis:
    logger.info("Received /analyze request")
    return await analyze_text_async(request.text)


@router.post(
    "/summarize",
    response_model=SummaryResponse,
    summary="Summarize text",
)
async def summarize(request: TextRequest) -> SummaryResponse:
    logger.info("Received /summarize request")
    summary = await summarize_text_async(request.text)
    return SummaryResponse(summary=summary)


@router.post(
    "/classify",
    response_model=ClassificationResponse,
    summary="Classify text",
)
async def classify(request: TextRequest) -> ClassificationResponse:
    logger.info("Received /classify request")
    category = await classify_text_async(request.text)
    return ClassificationResponse(category=category)


@router.post(
    "/extract-user",
    response_model=UserExtractResult,
    summary="Extract user info",
)
async def extract_user_endpoint(request: TextRequest) -> UserExtractResult:
    logger.info("Received /extract-user request")
    return await extract_user_async(request.text)


@router.post(
    "/analyze-many",
    response_model=AnalyzeManyResponse,
    summary="Analyze many texts",
)
async def analyze_many_endpoint(request: AnalyzeManyRequest) -> AnalyzeManyResponse:
    logger.info("Received /analyze-many request with %s texts", len(request.texts))
    results = await analyze_many_async(request.texts)
    return AnalyzeManyResponse(results=results)


@router.post(
    "/analyze-many-safe",
    response_model=AnalyzeManySafeResponse,
    summary="Analyze many texts safely",
)
async def analyze_many_safe_endpoint(
    request: AnalyzeManyRequest,
) -> AnalyzeManySafeResponse:
    logger.info("Received /analyze-many-safe request with %s texts", len(request.texts))
    results = await analyze_many_safe_async(request.texts)
    return AnalyzeManySafeResponse(results=results)