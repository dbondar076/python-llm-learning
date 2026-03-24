import asyncio

from app.models import ClassificationLabel, SafeTextAnalysis, TextAnalysis, UserExtractResult
from app.services.llm_service import (
    analyze_with_llm,
    analyze_with_llm_async,
    analyze_with_llm_safe_async,
    classify_with_llm,
    classify_with_llm_async,
    extract_user_with_llm,
    extract_user_with_llm_async,
    summarize_with_llm,
    summarize_with_llm_async,
)


def summarize_text(text: str) -> str:
    return summarize_with_llm(text).summary


async def summarize_text_async(text: str) -> str:
    return (await summarize_with_llm_async(text)).summary


def classify_text(text: str) -> ClassificationLabel:
    return classify_with_llm(text).label


async def classify_text_async(text: str) -> ClassificationLabel:
    return (await classify_with_llm_async(text)).label


def extract_user(text: str) -> UserExtractResult:
    return extract_user_with_llm(text)


async def extract_user_async(text: str) -> UserExtractResult:
    return await extract_user_with_llm_async(text)


def analyze_text(text: str) -> TextAnalysis:
    return analyze_with_llm(text)


async def analyze_text_async(text: str) -> TextAnalysis:
    return await analyze_with_llm_async(text)


async def analyze_text_safe_async(text: str) -> SafeTextAnalysis:
    return await analyze_with_llm_safe_async(text)


def analyze_many(texts: list[str]) -> list[TextAnalysis]:
    return [analyze_text(text) for text in texts]


async def analyze_many_async(texts: list[str]) -> list[TextAnalysis]:
    tasks = [analyze_text_async(text) for text in texts]
    return await asyncio.gather(*tasks)


async def analyze_many_safe_async(texts: list[str]) -> list[SafeTextAnalysis]:
    tasks = [analyze_text_safe_async(text) for text in texts]
    return await asyncio.gather(*tasks)