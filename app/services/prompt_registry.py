from app.settings import SUMMARY_PROMPT_VERSION


def get_active_summary_prompt_builder():
    if SUMMARY_PROMPT_VERSION not in SUMMARY_PROMPTS:
        raise ValueError(
            f"Unknown SUMMARY_PROMPT_VERSION: {SUMMARY_PROMPT_VERSION}"
        )
    return SUMMARY_PROMPTS[SUMMARY_PROMPT_VERSION]


def build_summary_prompt_v1(text: str) -> str:
    return (
        "Summarize the user's text.\n"
        "Return only the summary.\n"
        "The summary must be very short.\n"
        "The summary must contain at most 8 words.\n"
        "Do not add explanations or extra commentary.\n\n"
        f"Text:\n{text}"
    )


def build_summary_prompt_v2(text: str) -> str:
    return (
        "Write a very short summary of the user's text.\n"
        "Use at most 6 words.\n"
        "Do not repeat the full original sentence.\n"
        "Do not explain anything.\n"
        "Return only the summary.\n\n"
        f"Text:\n{text}"
    )


SUMMARY_PROMPTS = {
    "v1": build_summary_prompt_v1,
    "v2": build_summary_prompt_v2,
}