import json

from pydantic import ValidationError

from app.models import (
    TextAnalysis,
    UserExtractResult,
)
from app.services.llm_errors import (
    LLMExtractionError,
)

INVALID_NAMES = {"unknown", "n/a", "none", "null", "undefined", "-"}


def parse_text_analysis_json(text: str, raw_json: str) -> TextAnalysis:
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from LLM: {raw_json}") from e

    return TextAnalysis(
        text=text,
        category=data["category"],
        summary=normalize_summary(data["summary"]),
    )


def parse_user_extract_json(raw_json: str) -> UserExtractResult:
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise LLMExtractionError(f"Invalid JSON from LLM: {raw_json}") from e

    try:
        result = UserExtractResult(**data)
    except ValidationError as e:
        raise LLMExtractionError(f"Invalid extracted user data: {data}") from e

    name = result.name.strip().lower()
    age = result.age

    if name in INVALID_NAMES:
        raise LLMExtractionError(f"Invalid extracted user name: {result.name}")

    if age < 1 or age > 120:
        raise LLMExtractionError(f"Invalid extracted user age: {result.age}")

    return result


def normalize_summary(summary: str) -> str:
    summary = " ".join(summary.strip().split())
    if len(summary) > 80:
        summary = summary[:80].rstrip() + "..."
    return summary


def normalize_label(value: str) -> str:
    return value.strip().lower().rstrip(".")