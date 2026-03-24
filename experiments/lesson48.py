import asyncio
import json

from app.services.llm_parsers import normalize_summary
from app.services.llm_service import run_text_prompt_with_retry_async
from app.services.prompt_registry import SUMMARY_PROMPTS


def load_eval_cases(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_judge_prompt(text: str, summary_a: str, summary_b: str) -> str:
    return (
        "You are evaluating two summaries of the same text.\n"
        "Choose which summary is better.\n"
        "Criteria:\n"
        "- shorter is better if meaning is preserved\n"
        "- avoid repeating the original sentence exactly\n"
        "- keep the meaning accurate\n"
        "- return only one label: A or B\n\n"
        f"Original text:\n{text}\n\n"
        f"Summary A:\n{summary_a}\n\n"
        f"Summary B:\n{summary_b}\n"
    )


def normalize_judge_result(result: str) -> str:
    return result.strip().upper().rstrip(".")


async def generate_summary(version: str, text: str) -> str:
    prompt_builder = SUMMARY_PROMPTS[version]
    prompt = prompt_builder(text)
    raw_result = await run_text_prompt_with_retry_async(prompt)
    return normalize_summary(raw_result)


async def judge_pair(text: str, summary_a: str, summary_b: str) -> str:
    judge_prompt = build_judge_prompt(text, summary_a, summary_b)
    raw_result = await run_text_prompt_with_retry_async(judge_prompt)
    return normalize_judge_result(raw_result)


async def main() -> None:
    test_cases = load_eval_cases("../eval_cases.json")

    wins = {"v1": 0, "v2": 0}

    for case in test_cases:
        text = case["input"]

        summary_v1 = await generate_summary("v1", text)
        summary_v2 = await generate_summary("v2", text)

        winner = await judge_pair(text, summary_v1, summary_v2)

        print("TEXT:", text)
        print("V1:", summary_v1)
        print("V2:", summary_v2)
        print("JUDGE WINNER:", winner)
        print("-" * 50)

        if winner == "A":
            wins["v1"] += 1
        elif winner == "B":
            wins["v2"] += 1

    print("FINAL WINS:", wins)


if __name__ == "__main__":
    asyncio.run(main())