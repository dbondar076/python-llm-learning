import pytest
import app.settings as settings

from app.services.analyzer import analyze_text_safe_async
from app.services.llm_service import get_backoff_delay, reset_runtime_state

pytestmark = [pytest.mark.integration, pytest.mark.mock_only]


def test_get_backoff_delay() -> None:
    assert get_backoff_delay(1) == 0.5
    assert get_backoff_delay(2) == 1.0
    assert get_backoff_delay(3) == 2.0


@pytest.fixture(autouse=True)
def setup_mock_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "USE_REAL_LLM", False)
    monkeypatch.setattr(settings, "LLM_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(settings, "LLM_MAX_RETRIES", 2)
    reset_runtime_state()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_analyze_text_safe_async_success() -> None:
    result = await analyze_text_safe_async("What is Python?")

    assert result.degraded is False
    assert result.category == "question"
    assert isinstance(result.summary, str)
    assert result.summary.strip() != ""


@pytest.mark.integration
@pytest.mark.asyncio
async def test_analyze_text_safe_async_timeout_degraded() -> None:
    result = await analyze_text_safe_async("TIMEOUT: What is Python?")

    assert result.degraded is True
    assert result.category is None
    assert isinstance(result.summary, str)
    assert result.summary.strip() != ""