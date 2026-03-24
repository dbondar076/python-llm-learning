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


if __name__ == "__main__":
    result = analyze_all(texts)
    print(result)