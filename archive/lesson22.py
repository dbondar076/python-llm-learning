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


if __name__ == "__main__":
    result = analyze_all(texts)
    print(result)

    for item in result:
        print(item.text, item.category, item.summary)


# from typing import Literal
# from pydantic import BaseModel, Field
#
#
# Category = Literal["question", "statement", "short"]
#
#
# class TextAnalysis(BaseModel):
#     text: str = Field(min_length=1)
#     category: Category
#     summary: str = Field(min_length=1)