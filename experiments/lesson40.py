import asyncio

from app.services.llm_service import analyze_with_llm_async


test_cases = [
    {
        "input": "What is Python?",
        "expected_category": "question",
        "acceptable_summaries": {
            "asking what python is",
            "request definition of python",
            "question about python",
            "asking definition of python",
        },
    },
    {
        "input": "Hi",
        "expected_category": "short",
        "acceptable_summaries": {
            "brief greeting",
            "simple greeting",
            "a greeting",
            "user says hi",
        },
    },
    {
        "input": "Python is a programming language.",
        "expected_category": "statement",
        "acceptable_summaries": {
            "python: a programming language",
            "python programming language",
            "statement about python language",
            "says python is a programming language",
        },
    },
]


def normalize_eval_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


async def run_eval() -> None:
    passed_category = 0
    passed_summary = 0
    total = len(test_cases)

    for case in test_cases:
        result = await analyze_with_llm_async(case["input"])

        category_ok = result.category == case["expected_category"]
        normalized_summary = normalize_eval_text(result.summary)
        summary_ok = normalized_summary in case["acceptable_summaries"]

        print("INPUT:", case["input"])
        print("EXPECTED CATEGORY:", case["expected_category"])
        print("ACTUAL CATEGORY:", result.category)
        print("CATEGORY PASS:", category_ok)
        print("SUMMARY:", result.summary)
        print("NORMALIZED SUMMARY:", normalized_summary)
        print("SUMMARY PASS:", summary_ok)
        print("-" * 40)

        if category_ok:
            passed_category += 1
        if summary_ok:
            passed_summary += 1

    print(f"Category score: {passed_category}/{total}")
    print(f"Golden summary score: {passed_summary}/{total}")


if __name__ == "__main__":
    asyncio.run(run_eval())