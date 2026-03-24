from lesson18 import ask_llm_http


def summarize_text(text: str) -> dict | None:
    prompt = f"Summarize this text in one short sentence:\n\n{text}"
    return ask_llm_http(prompt)


def classify_text(text: str) -> dict | None:
    prompt = f"Classify this text as question, statement, or short:\n\n{text}"
    return ask_llm_http(prompt)


if __name__ == "__main__":
    summary_result = summarize_text("Python is a popular programming language used in AI.")
    classification_result = classify_text("What is Python?")

    print(summary_result)
    print(classification_result)