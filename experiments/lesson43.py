import asyncio
import json

from app.services.llm_parsers import normalize_summary
from app.services.llm_service import run_text_prompt_with_retry_async
from app.services.prompt_registry import SUMMARY_PROMPTS


def normalize_eval_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def load_eval_cases(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def check_summary_by_rules(summary: str, rules: dict) -> bool:
    normalized = normalize_eval_text(summary)

    for item in rules.get("must_contain", []):
        if item not in normalized:
            return False

    one_of = rules.get("must_contain_one_of", [])
    if one_of and not any(item in normalized for item in one_of):
        return False

    return True


async def evaluate_summary_prompt(version: str, test_cases: list[dict]) -> None:
    prompt_builder = SUMMARY_PROMPTS[version]

    passed = 0
    total = len(test_cases)

    print(f"\n=== Evaluating summary prompt {version} ===")

    for case in test_cases:
        prompt = prompt_builder(case["input"])
        raw_result = await run_text_prompt_with_retry_async(prompt)
        summary = normalize_summary(raw_result)

        summary_ok = check_summary_by_rules(summary, case["summary_rules"])

        print("INPUT:", case["input"])
        print("SUMMARY:", summary)
        print("PASS:", summary_ok)
        print("-" * 40)

        if summary_ok:
            passed += 1

    print(f"Prompt {version} score: {passed}/{total}")


async def main() -> None:
    test_cases = load_eval_cases("../eval_cases.json")

    for version in SUMMARY_PROMPTS:
        await evaluate_summary_prompt(version, test_cases)


if __name__ == "__main__":
    asyncio.run(main())