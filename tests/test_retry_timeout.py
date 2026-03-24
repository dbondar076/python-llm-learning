import pytest
import app.settings as settings

from app.services.llm_errors import LLMTimeoutError
from app.services.llm_service import analyze_with_llm_async, reset_runtime_state

pytestmark = [pytest.mark.integration, pytest.mark.mock_only]


@pytest.fixture(autouse=True)
def setup_mock_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "USE_REAL_LLM", False)
    monkeypatch.setattr(settings, "LLM_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(settings, "LLM_MAX_RETRIES", 2)
    reset_runtime_state()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_analyze_with_llm_async_success() -> None:
    result = await analyze_with_llm_async("What is Python?")

    assert result.category == "question"
    assert isinstance(result.summary, str)
    assert result.summary.strip() != ""


@pytest.mark.integration
@pytest.mark.asyncio
async def test_analyze_with_llm_async_retry() -> None:
    result = await analyze_with_llm_async("RETRY: What is Python?")

    assert result.text == "RETRY: What is Python?"
    assert result.category == "question"
    assert isinstance(result.summary, str)
    assert result.summary.strip() != ""


@pytest.mark.integration
@pytest.mark.asyncio
async def test_analyze_with_llm_async_timeout() -> None:
    with pytest.raises(LLMTimeoutError):
        await analyze_with_llm_async("TIMEOUT: What is Python?")