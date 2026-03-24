import asyncio

from app.services.llm_service import analyze_with_llm_async


test_cases = [
    {
        "input": "What is Python?",
        "expected_category": "question",
    },
    {
        "input": "Hi",
        "expected_category": "short",
    },
    {
        "input": "Python is a programming language.",
        "expected_category": "statement",
    },
]


def check_summary(summary: str, original_text: str) -> tuple[bool, list[str]]:
    errors: list[str] = []

    if not summary.strip():
        errors.append("summary is empty")

    if len(summary) > 80:
        errors.append("summary is too long in characters")

    if len(summary.split()) > 8:
        errors.append("summary is too long in words")

    if "\n" in summary:
        errors.append("summary contains newline")

    if summary.strip() == original_text.strip():
        errors.append("summary equals original text")

    return len(errors) == 0, errors


async def run_eval() -> None:
    passed_category = 0
    passed_summary = 0
    total = len(test_cases)

    for case in test_cases:
        result = await analyze_with_llm_async(case["input"])

        category_ok = result.category == case["expected_category"]
        summary_ok, summary_errors = check_summary(result.summary, case["input"])

        print("INPUT:", case["input"])
        print("EXPECTED CATEGORY:", case["expected_category"])
        print("ACTUAL CATEGORY:", result.category)
        print("CATEGORY PASS:", category_ok)
        print("SUMMARY:", result.summary)
        print("SUMMARY PASS:", summary_ok)

        if not summary_ok:
            print("SUMMARY ERRORS:", summary_errors)

        print("-" * 40)

        if category_ok:
            passed_category += 1
        if summary_ok:
            passed_summary += 1

    print(f"Category score: {passed_category}/{total}")
    print(f"Summary score: {passed_summary}/{total}")


if __name__ == "__main__":
    asyncio.run(run_eval())