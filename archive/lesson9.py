import json
from typing import TypedDict


class TextStats(TypedDict):
    text: str
    length: int
    word_count: int
    has_question: bool
    is_long: bool


texts = [
    "What is Python?",
    "Python is a programming language.",
    "Hi",
    "How do LLMs work?",
    "OK",
    "LLMs work with text data",
]


def analyze_text(text: str) -> TextStats:
    return {
        "text": text,
        "length": len(text),
        "word_count": len(text.split()),
        "has_question": "?" in text,
        "is_long": len(text) > 20,
    }


def analyze_all(text_fragments: list[str]) -> list[TextStats]:
    return [analyze_text(text) for text in text_fragments]


def filter_useful_texts(stats_list: list[TextStats]) -> list[TextStats]:
    return [item for item in stats_list if item["word_count"] >= 3]


def save_results(results: list[TextStats], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    analyzed = analyze_all(texts)
    useful = filter_useful_texts(analyzed)
    save_results(useful, "useful_texts.json")
    print(useful)