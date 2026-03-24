PROMPT_PREFIX = "Analyze the user's text"


def build_summary_prompt(text: str) -> str:
    return (
        "Summarize the user's text.\n"
        "Return only the summary.\n"
        "The summary must be very short.\n"
        "The summary must contain at most 8 words.\n"
        "Do not add explanations or extra commentary.\n\n"
        f"Text:\n{text}"
    )


def build_classification_prompt(text: str) -> str:
    return (
        "Classify the user's text.\n"
        "Return exactly one label: question, statement, or short.\n"
        "Do not add anything else.\n\n"
        f"Text:\n{text}"
    )


def build_analysis_prompt(text: str) -> str:
    return (
        f"{PROMPT_PREFIX}.\n"
        "Return JSON that matches the schema.\n"
        "Rules:\n"
        "- category must be one of: question, statement, short\n"
        "- summary must be very short\n"
        "- summary must contain at most 8 words\n"
        "- summary must not include explanations, ambiguity notes, or extra commentary\n"
        "- summary should describe the user's text, not answer it\n\n"
        f"Text:\n{text}"
    )


def build_user_extraction_prompt(text: str) -> str:
    return (
        "Extract user name and age from the text.\n"
        "Return JSON that matches the schema.\n"
        "Do not invent or guess values.\n"
        "If the text does not explicitly contain both a valid name and a valid age, return invalid data.\n"
        "Do not use placeholders like 'unknown', 'n/a', 'none', or 999.\n\n"
        f"Text:\n{text}"
    )
