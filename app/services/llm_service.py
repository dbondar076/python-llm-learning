import asyncio
import json
import logging
import app.settings as settings

from app.models import (
    ClassificationResult,
    SafeTextAnalysis,
    SummaryResult,
    TextAnalysis,
    UserExtractResult,
)

from app.services.llm_cache import (
    JSON_CACHE,
    TEXT_CACHE,
    get_cached_value,
    set_cached_value,
)

from app.services.llm_errors import (
    LLMExtractionError,
    LLMRetryError,
    LLMTimeoutError,
)

from app.services.llm_prompts import (
    PROMPT_PREFIX,
    build_analysis_prompt,
    build_classification_prompt,
    build_user_extraction_prompt,
)

from app.services.llm_schemas import (
    build_analysis_json_schema,
    build_user_extraction_json_schema,
)

from app.services.llm_parsers import (
    normalize_label,
    normalize_summary,
    parse_text_analysis_json,
    parse_user_extract_json,
)

from app.services.prompt_registry import get_active_summary_prompt_builder
from app.services.openai_client import reset_openai_client, get_openai_client

logger = logging.getLogger(__name__)

FLAKY_ATTEMPTS: dict[str, int] = {}

_LLM_SEMAPHORE: asyncio.Semaphore | None = None
_LLM_SEMAPHORE_LIMIT: int | None = None


def get_llm_semaphore() -> asyncio.Semaphore:
    global _LLM_SEMAPHORE, _LLM_SEMAPHORE_LIMIT

    current_limit = settings.LLM_CONCURRENCY_LIMIT

    if current_limit <= 0:
        raise ValueError("LLM_CONCURRENCY_LIMIT must be greater than 0")

    if _LLM_SEMAPHORE is None or _LLM_SEMAPHORE_LIMIT != current_limit:
        _LLM_SEMAPHORE = asyncio.Semaphore(current_limit)
        _LLM_SEMAPHORE_LIMIT = current_limit

    return _LLM_SEMAPHORE


def get_backoff_delay(attempt: int) -> float:
    return settings.LLM_BASE_DELAY_SECONDS * (2 ** (attempt - 1))


def reset_runtime_state() -> None:
    global _LLM_SEMAPHORE, _LLM_SEMAPHORE_LIMIT

    reset_openai_client()
    _LLM_SEMAPHORE = None
    _LLM_SEMAPHORE_LIMIT = None
    FLAKY_ATTEMPTS.clear()


# ----------------------------
# MOCK / LOCAL IMPLEMENTATION
# ----------------------------

def run_llm_prompt(prompt: str) -> str:
    if prompt.startswith("Summarize the user's text"):
        text = prompt.split("Text:\n", 1)[1]
        return normalize_summary(text if len(text) <= 25 else f"{text[:25]}...")

    if prompt.startswith("Classify the user's text"):
        text = prompt.split("Text:\n", 1)[1]

        if "?" in text:
            return "question"
        if len(text.split()) < 3:
            return "short"
        return "statement"

    if prompt.startswith(PROMPT_PREFIX):
        text = prompt.split("Text:\n", 1)[1]

        if "?" in text:
            category = "question"
        elif len(text.split()) < 3:
            category = "short"
        else:
            category = "statement"

        summary = text if len(text) <= 25 else f"{text[:25]}..."
        return f"{category}|{normalize_summary(summary)}"

    if prompt.startswith("Extract user name and age"):
        text = prompt.split("Text:\n", 1)[1]

        if "," not in text or ":" not in text:
            raise LLMExtractionError(f"Invalid input format: {text}")

        try:
            name_part, age_part = text.split(",")
            name = name_part.split(":")[1].strip()
            age = age_part.split(":")[1].strip()
            return f"{name}|{age}"
        except Exception as e:
            raise LLMExtractionError(f"Failed to parse user data: {text}") from e

    raise ValueError("Unsupported prompt")


async def run_text_prompt_async_flaky(prompt: str) -> str:
    async with get_llm_semaphore():
        count = FLAKY_ATTEMPTS.get(prompt, 0)
        FLAKY_ATTEMPTS[prompt] = count + 1

        logger.info("Attempt %s for prompt", FLAKY_ATTEMPTS[prompt])

        if "RETRY" in prompt and count == 0:
            await asyncio.sleep(0.2)
            raise ValueError("Temporary LLM failure")

        if "TIMEOUT" in prompt:
            await asyncio.sleep(settings.LLM_TIMEOUT_SECONDS + 1)
            return run_llm_prompt(prompt)

        await asyncio.sleep(0.05)
        return run_llm_prompt(prompt)


async def run_json_prompt_async_flaky(prompt: str, schema: dict) -> str:
    del schema  # not used in mock mode

    async with get_llm_semaphore():
        count = FLAKY_ATTEMPTS.get(prompt, 0)
        FLAKY_ATTEMPTS[prompt] = count + 1

        logger.info("Attempt %s for prompt", FLAKY_ATTEMPTS[prompt])

        if "RETRY" in prompt and count == 0:
            await asyncio.sleep(0.2)
            raise ValueError("Temporary LLM failure")

        if "TIMEOUT" in prompt:
            await asyncio.sleep(settings.LLM_TIMEOUT_SECONDS + 1)

        if prompt.startswith(PROMPT_PREFIX):
            text = prompt.split("Text:\n", 1)[1]

            if "?" in text:
                category = "question"
            elif len(text.split()) < 3:
                category = "short"
            else:
                category = "statement"

            summary = text if len(text) <= 25 else f"{text[:25]}..."
            return json.dumps(
                {
                    "category": category,
                    "summary": normalize_summary(summary),
                }
            )

        if prompt.startswith("Extract user name and age"):
            text = prompt.split("Text:\n", 1)[1]

            if "," not in text or ":" not in text:
                raise LLMExtractionError(f"Invalid input format: {text}")

            try:
                name_part, age_part = text.split(",")
                name = name_part.split(":")[1].strip()
                age = int(age_part.split(":")[1].strip())
                return json.dumps({"name": name, "age": age})
            except Exception as e:
                raise LLMExtractionError(f"Failed to parse user data: {text}") from e

        raise ValueError("Unsupported JSON mock prompt")


# ----------------------------
# REAL OPENAI IMPLEMENTATION
# ----------------------------

def run_text_prompt_real(prompt: str) -> str:
    cached = get_cached_value(TEXT_CACHE, prompt)
    if cached is not None:
        logger.info("TEXT cache hit")
        return cached

    logger.info("TEXT cache miss")
    client = get_openai_client()

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
    )

    result = response.output_text
    set_cached_value(TEXT_CACHE, prompt, result)
    return result


def run_json_prompt_real(prompt: str, schema: dict) -> str:
    cached = get_cached_value(JSON_CACHE, prompt)
    if cached is not None:
        logger.info("JSON cache hit")
        return cached

    logger.info("JSON cache miss")

    client = get_openai_client()

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
        text={"format": schema},
    )

    result = response.output_text
    set_cached_value(JSON_CACHE, prompt, result)
    return result


async def run_text_prompt_async_real(prompt: str) -> str:
    import time

    cached = get_cached_value(TEXT_CACHE, prompt)
    if cached is not None:
        logger.info("TEXT cache hit")
        return cached

    logger.info("TEXT cache miss")
    client = get_openai_client()

    async with get_llm_semaphore():
        logger.info("LLM REAL text call")

        start = time.perf_counter()

        response = await asyncio.to_thread(
            client.responses.create,
            model="gpt-5-mini",
            input=prompt,
        )

        elapsed = time.perf_counter() - start
        logger.info("LLM REAL text call finished in %.2fs", elapsed)

        result = response.output_text
        set_cached_value(TEXT_CACHE, prompt, result)
        return result


async def run_json_prompt_async_real(prompt: str, schema: dict) -> str:
    import time

    cached = get_cached_value(JSON_CACHE, prompt)
    if cached is not None:
        logger.info("JSON cache hit")
        return cached

    logger.info("JSON cache miss")
    client = get_openai_client()

    async with get_llm_semaphore():
        logger.info("LLM REAL json call")

        start = time.perf_counter()

        response = await asyncio.to_thread(
            client.responses.create,
            model="gpt-5-mini",
            input=prompt,
            text={"format": schema},
        )

        elapsed = time.perf_counter() - start
        logger.info("LLM REAL json call finished in %.2fs", elapsed)

        result = response.output_text
        set_cached_value(JSON_CACHE, prompt, result)
        return result


# ----------------------------
# RETRY / TIMEOUT WRAPPERS
# ----------------------------

async def run_text_prompt_with_retry_async(prompt: str) -> str:
    last_error: Exception | None = None

    for attempt in range(1, settings.LLM_MAX_RETRIES + 1):
        try:
            logger.info("Running text attempt %s/%s", attempt, settings.LLM_MAX_RETRIES)

            runner = (
                run_text_prompt_async_real(prompt)
                if settings.USE_REAL_LLM
                else run_text_prompt_async_flaky(prompt)
            )

            return await asyncio.wait_for(
                runner,
                timeout=settings.LLM_TIMEOUT_SECONDS,
            )

        except asyncio.TimeoutError as e:
            logger.warning("Timeout on text attempt %s", attempt)
            last_error = e

        except ValueError as e:
            logger.warning("Retryable text error on attempt %s: %s", attempt, e)
            last_error = e

        if attempt < settings.LLM_MAX_RETRIES:
            delay = get_backoff_delay(attempt)
            logger.info("Backing off for %.1fs", delay)
            await asyncio.sleep(delay)

    if isinstance(last_error, asyncio.TimeoutError):
        raise LLMTimeoutError(
            f"LLM text request timed out after {settings.LLM_MAX_RETRIES} attempts"
        ) from last_error

    raise LLMRetryError(
        f"LLM text request failed after {settings.LLM_MAX_RETRIES} attempts"
    ) from last_error


async def run_json_prompt_with_retry_async(prompt: str, schema: dict) -> str:
    last_error: Exception | None = None

    for attempt in range(1, settings.LLM_MAX_RETRIES + 1):
        try:
            logger.info("Running json attempt %s/%s", attempt, settings.LLM_MAX_RETRIES)

            runner = (
                run_json_prompt_async_real(prompt, schema)
                if settings.USE_REAL_LLM
                else run_json_prompt_async_flaky(prompt, schema)
            )

            return await asyncio.wait_for(
                runner,
                timeout=settings.LLM_TIMEOUT_SECONDS,
            )

        except asyncio.TimeoutError as e:
            logger.warning("Timeout on json attempt %s", attempt)
            last_error = e

        except ValueError as e:
            logger.warning("Retryable json error on attempt %s: %s", attempt, e)
            last_error = e

        if attempt < settings.LLM_MAX_RETRIES:
            delay = get_backoff_delay(attempt)
            logger.info("Backing off for %.1fs", delay)
            await asyncio.sleep(delay)

    if isinstance(last_error, asyncio.TimeoutError):
        raise LLMTimeoutError(
            f"LLM json request timed out after {settings.LLM_MAX_RETRIES} attempts"
        ) from last_error

    raise LLMRetryError(
        f"LLM json request failed after {settings.LLM_MAX_RETRIES} attempts"
    ) from last_error


# ----------------------------
# PUBLIC SERVICE FUNCTIONS
# ----------------------------

def summarize_with_llm(text: str) -> SummaryResult:
    prompt_builder = get_active_summary_prompt_builder()
    prompt = prompt_builder(text)

    if settings.USE_REAL_LLM:
        raw_result = run_text_prompt_real(prompt)
        return SummaryResult(summary=normalize_summary(raw_result))

    result = run_llm_prompt(prompt)
    return SummaryResult(summary=normalize_summary(result))


async def summarize_with_llm_async(text: str) -> SummaryResult:
    prompt_builder = get_active_summary_prompt_builder()
    prompt = prompt_builder(text)

    raw_result = await run_text_prompt_with_retry_async(prompt)
    return SummaryResult(summary=normalize_summary(raw_result))


def classify_with_llm(text: str) -> ClassificationResult:
    prompt = build_classification_prompt(text)

    if settings.USE_REAL_LLM:
        raw_result = run_text_prompt_real(prompt)
        result = normalize_label(raw_result)

        if result not in ("question", "statement", "short"):
            raise ValueError(f"Invalid classification result: {result}")

        return ClassificationResult(label=result)

    result = run_llm_prompt(prompt)

    if result not in ("question", "statement", "short"):
        raise ValueError(f"Invalid classification result: {result}")

    return ClassificationResult(label=result)


async def classify_with_llm_async(text: str) -> ClassificationResult:
    prompt = build_classification_prompt(text)

    raw_result = await run_text_prompt_with_retry_async(prompt)
    result = normalize_label(raw_result)

    if result not in ("question", "statement", "short"):
        raise ValueError(f"Invalid classification result: {result}")

    return ClassificationResult(label=result)


def analyze_with_llm(text: str) -> TextAnalysis:
    prompt = build_analysis_prompt(text)

    if settings.USE_REAL_LLM:
        raw_result = run_json_prompt_real(prompt, build_analysis_json_schema())
        return parse_text_analysis_json(text, raw_result)

    raw_result = run_llm_prompt(prompt)
    category, summary = raw_result.split("|", 1)

    return TextAnalysis(
        text=text,
        category=category,
        summary=normalize_summary(summary),
    )


async def analyze_with_llm_async(text: str) -> TextAnalysis:
    prompt = build_analysis_prompt(text)

    raw_result = await run_json_prompt_with_retry_async(
        prompt,
        build_analysis_json_schema(),
    )
    return parse_text_analysis_json(text, raw_result)


async def analyze_with_llm_safe_async(text: str) -> SafeTextAnalysis:
    try:
        result = await analyze_with_llm_async(text)
        return SafeTextAnalysis(
            text=result.text,
            category=result.category,
            summary=result.summary,
            degraded=False,
        )
    except (LLMTimeoutError, LLMRetryError):
        fallback_summary = text if len(text) <= 25 else f"{text[:25]}..."
        logger.warning("Falling back to degraded analysis for text: %s", text)
        return SafeTextAnalysis(
            text=text,
            category=None,
            summary=normalize_summary(fallback_summary),
            degraded=True,
        )


def extract_user_with_llm(text: str) -> UserExtractResult:
    prompt = build_user_extraction_prompt(text)

    if settings.USE_REAL_LLM:
        raw_result = run_json_prompt_real(prompt, build_user_extraction_json_schema())
        return parse_user_extract_json(raw_result)

    try:
        result = run_llm_prompt(prompt)
        name, age = result.split("|", 1)
        return UserExtractResult(name=name, age=int(age))
    except (ValueError, IndexError) as e:
        raise LLMExtractionError(f"Failed to extract user from text: {text}") from e


async def extract_user_with_llm_async(text: str) -> UserExtractResult:
    prompt = build_user_extraction_prompt(text)

    raw_result = await run_json_prompt_with_retry_async(
        prompt,
        build_user_extraction_json_schema(),
    )
    return parse_user_extract_json(raw_result)