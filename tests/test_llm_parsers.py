import pytest

from app.services.llm_errors import LLMExtractionError
from app.services.llm_parsers import (
    normalize_label,
    normalize_summary,
    parse_text_analysis_json,
    parse_user_extract_json,
)

pytestmark = pytest.mark.fast


def test_normalize_summary() -> None:
    summary = "  Python   is \n great  "
    assert normalize_summary(summary) == "Python is great"


def test_normalize_label() -> None:
    assert normalize_label(" Question.\n") == "question"


def test_parse_text_analysis_json() -> None:
    raw_json = '{"category":"question","summary":"Asking what Python is"}'
    result = parse_text_analysis_json("What is Python?", raw_json)

    assert result.text == "What is Python?"
    assert result.category == "question"
    assert result.summary == "Asking what Python is"


def test_parse_user_extract_json() -> None:
    raw_json = '{"name":"Dima","age":30}'
    result = parse_user_extract_json(raw_json)

    assert result.name == "Dima"
    assert result.age == 30


def test_parse_user_extract_json_invalid_json() -> None:
    with pytest.raises(LLMExtractionError):
        parse_user_extract_json("not-json")