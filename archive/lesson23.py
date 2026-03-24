import json
from typing import Literal

from pydantic import BaseModel


class TextAnalysis(BaseModel):
    text: str
    category: Literal["question", "statement", "short"]
    summary: str


texts = [
    "What is Python?",
    "Python is a popular programming language used in AI.",
    "Hi",
]


def detect_category(text: str) -> Literal["question", "statement", "short"]:
    if "?" in text:
        return "question"
    elif len(text.split()) < 3:
        return "short"
    else:
        return "statement"


def build_summary(text: str) -> str:
    return text if len(text) <= 25 else f"{text[:25]}..."


def analyze_text(text: str) -> TextAnalysis:
    return TextAnalysis(
        text=text,
        category=detect_category(text),
        summary=build_summary(text),
    )


def analyze_all(texts: list[str]) -> list[TextAnalysis]:
    return [analyze_text(text) for text in texts]


def save_results(results: list[TextAnalysis], file_path: str) -> None:
    data = [item.model_dump() for item in results]
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    result = analyze_all(texts)
    save_results(result, "text_analysis.json")
    print(result)