import asyncio

from app.services.llm_service import run_text_prompt_with_retry_async
from app.services.llm_parsers import normalize_summary


test_cases = [
    "What is Python?",
    "Hi",
    "Python is a programming language.",
    "FastAPI is a modern Python framework for building APIs.",
]


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


def check_summary(summary: str, original_text: str) -> bool:
    if not summary.strip():
        return False
    if len(summary) > 80:
        return False
    if len(summary.split()) > 8:
        return False
    if "\n" in summary:
        return False
    if summary.strip() == original_text.strip():
        return False
    return True


async def evaluate_prompt(prompt_builder, label: str) -> None:
    passed = 0
    total = len(test_cases)

    print(f"\n=== Evaluating {label} ===")

    for text in test_cases:
        prompt = prompt_builder(text)
        raw_result = await run_text_prompt_with_retry_async(prompt)
        summary = normalize_summary(raw_result)

        ok = check_summary(summary, text)

        print("INPUT:", text)
        print("SUMMARY:", summary)
        print("PASS:", ok)
        print("-" * 40)

        if ok:
            passed += 1

    print(f"{label} score: {passed}/{total}")


async def main() -> None:
    await evaluate_prompt(build_summary_prompt_v1, "Prompt V1")
    await evaluate_prompt(build_summary_prompt_v2, "Prompt V2")


if __name__ == "__main__":
    asyncio.run(main())