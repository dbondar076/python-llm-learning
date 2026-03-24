import asyncio

from app.services.llm_service import run_text_prompt_with_retry_async


TEXT = "Python is a programming language."

SUMMARY_A = "Python is a programming language."
SUMMARY_B = "Python programming language"


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


async def main() -> None:
    prompt = build_judge_prompt(TEXT, SUMMARY_A, SUMMARY_B)
    raw_result = await run_text_prompt_with_retry_async(prompt)
    winner = normalize_judge_result(raw_result)

    print("TEXT:", TEXT)
    print("SUMMARY A:", SUMMARY_A)
    print("SUMMARY B:", SUMMARY_B)
    print("RAW RESULT:", raw_result)
    print("WINNER:", winner)


if __name__ == "__main__":
    asyncio.run(main())