import asyncio

from app.services.llm_parsers import normalize_summary
from app.services.llm_service import run_text_prompt_with_retry_async
from app.services.prompt_registry import SUMMARY_PROMPTS


test_cases = [
    "What is Python?",
    "Hi",
    "Python is a programming language.",
    "FastAPI is a modern Python framework for building APIs.",
]


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


async def evaluate_prompt(version: str) -> None:
    prompt_builder = SUMMARY_PROMPTS[version]

    passed = 0
    total = len(test_cases)

    print(f"\n=== Evaluating summary prompt {version} ===")

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

    print(f"Prompt {version} score: {passed}/{total}")


async def main() -> None:
    for version in SUMMARY_PROMPTS:
        await evaluate_prompt(version)


if __name__ == "__main__":
    asyncio.run(main())