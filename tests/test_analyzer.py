import pytest

from app.services.analyzer import (
    analyze_text,
    classify_text,
    summarize_text,
)

pytestmark = pytest.mark.fast


def test_classify_text_question() -> None:
    assert classify_text("What is Python?") == "question"


def test_classify_text_short() -> None:
    assert classify_text("Hi") == "short"


def test_classify_text_statement() -> None:
    assert classify_text("Python is a programming language.") == "statement"


def test_summarize_text_not_empty() -> None:
    summary = summarize_text("Python is a popular programming language used in AI.")

    assert isinstance(summary, str)
    assert summary.strip() != ""


def test_analyze_text_shape() -> None:
    result = analyze_text("What is Python?")

    assert result.text == "What is Python?"
    assert result.category == "question"
    assert isinstance(result.summary, str)
    assert result.summary.strip() != ""