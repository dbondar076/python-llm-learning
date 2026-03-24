import pytest
import app.settings as settings

from app.services.llm_service import (
    summarize_with_llm_async,
    classify_with_llm_async,
    analyze_with_llm_async,
)


pytestmark = [pytest.mark.real_llm, pytest.mark.asyncio]


@pytest.mark.skipif(not settings.OPENAI_API_KEY, reason="OPENAI_API_KEY is not set")
async def test_real_llm_summarize_smoke() -> None:
    result = await summarize_with_llm_async("Python is a programming language used for many tasks.")
    assert result.summary


@pytest.mark.skipif(not settings.OPENAI_API_KEY, reason="OPENAI_API_KEY is not set")
async def test_real_llm_classify_smoke() -> None:
    result = await classify_with_llm_async("What is Python?")
    assert result.label in ("question", "statement", "short")


@pytest.mark.skipif(not settings.OPENAI_API_KEY, reason="OPENAI_API_KEY is not set")
async def test_real_llm_analyze_smoke() -> None:
    result = await analyze_with_llm_async("What is Python?")
    assert result.category in ("question", "statement", "short")
    assert result.summary