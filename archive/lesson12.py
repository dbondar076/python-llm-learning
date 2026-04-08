from typing import Literal, TypedDict


class TextAnalysis(TypedDict):
    text: str
    category: Literal["question", "statement", "short"]
    summary: str


texts = [
    "What is Python?",
    "Python is a popular programming language used in AI.",
    "Hi",
    "How do LLMs work?",
    "LLMs can summarize text.",
]


def detect_category(text: str) -> Literal["question", "statement", "short"]:
    if "?" in text:
        return "question"
    elif len(text.split()) < 3:
        return "short"
    else:
        return "statement"


def build_summary(text: str) -> str:
    return text if len(text) <= 25 else text[:25] + "..."


def analyze_text(text: str) -> TextAnalysis:
    return {
        "text": text,
        "category": detect_category(text),
        "summary": build_summary(text),
    }


def analyze_all(texts: list[str]) -> list[TextAnalysis]:
    return [analyze_text(text) for text in texts]


if __name__ == "__main__":
    result = analyze_all(texts)
    print(result)


# from typing import Literal, TypedDict
#
#
# Category = Literal["question", "statement", "short"]
#
#
# class TextAnalysis(TypedDict):
#     text: str
#     category: Category
#     summary: str
#
#
# def detect_category(text: str) -> Category:
#     normalized = text.strip()
#
#     if not normalized:
#         return "short"
#
#     if normalized.endswith("?"):
#         return "question"
#
#     if len(normalized.split()) < 3:
#         return "short"
#
#     return "statement"
#
#
# def build_summary(text: str, max_length: int = 25) -> str:
#     normalized = text.strip()
#
#     if len(normalized) <= max_length:
#         return normalized
#
#     return normalized[:max_length].rstrip() + "..."
#
#
# def analyze_text(text: str) -> TextAnalysis:
#     return {
#         "text": text,
#         "category": detect_category(text),
#         "summary": build_summary(text),
#     }
#
#
# def analyze_all(texts: list[str]) -> list[TextAnalysis]:
#     return [analyze_text(text) for text in texts]