import pytest

from app.services.llm_prompts import (
    build_analysis_prompt,
    build_classification_prompt,
    build_summary_prompt,
    build_user_extraction_prompt,
)

pytestmark = pytest.mark.fast


def test_build_summary_prompt() -> None:
    text = "Python is great"
    prompt = build_summary_prompt(text)

    assert "Summarize the user's text." in prompt
    assert "Text:\nPython is great" in prompt


def test_build_classification_prompt() -> None:
    text = "What is Python?"
    prompt = build_classification_prompt(text)

    assert "Classify the user's text." in prompt
    assert "Return exactly one label" in prompt
    assert "Text:\nWhat is Python?" in prompt


def test_build_analysis_prompt() -> None:
    text = "What is Python?"
    prompt = build_analysis_prompt(text)

    assert "Analyze the user's text." in prompt
    assert "Return JSON that matches the schema." in prompt
    assert "Text:\nWhat is Python?" in prompt


def test_build_user_extraction_prompt() -> None:
    text = "Name: Dima, Age: 30"
    prompt = build_user_extraction_prompt(text)

    assert "Extract user name and age from the text." in prompt
    assert "Return JSON that matches the schema." in prompt
    assert "Text:\nName: Dima, Age: 30" in prompt