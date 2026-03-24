import asyncio

from app.services.llm_service import analyze_with_llm_async


test_cases = [
    {
        "input": "What is Python?",
        "expected_category": "question",
        "summary_rules": {
            "must_contain": ["python"],
            "must_contain_one_of": ["definition", "question", "asking", "request"],
        },
    },
    {
        "input": "Hi",
        "expected_category": "short",
        "summary_rules": {
            "must_contain_one_of": ["greeting", "hi"],
        },
    },
    {
        "input": "Python is a programming language.",
        "expected_category": "statement",
        "summary_rules": {
            "must_contain": ["python"],
            "must_contain_one_of": ["programming language", "language"],
        },
    },
]


def normalize_eval_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def check_summary_by_rules(summary: str, rules: dict) -> bool:
    normalized = normalize_eval_text(summary)

    for item in rules.get("must_contain", []):
        if item not in normalized:
            return False

    one_of = rules.get("must_contain_one_of", [])
    if one_of and not any(item in normalized for item in one_of):
        return False

    return True


async def run_eval() -> None:
    passed_category = 0
    passed_summary = 0
    total = len(test_cases)

    for case in test_cases:
        result = await analyze_with_llm_async(case["input"])

        category_ok = result.category == case["expected_category"]
        summary_ok = check_summary_by_rules(result.summary, case["summary_rules"])

        print("INPUT:", case["input"])
        print("EXPECTED CATEGORY:", case["expected_category"])
        print("ACTUAL CATEGORY:", result.category)
        print("CATEGORY PASS:", category_ok)
        print("SUMMARY:", result.summary)
        print("SUMMARY PASS:", summary_ok)
        print("-" * 40)

        if category_ok:
            passed_category += 1
        if summary_ok:
            passed_summary += 1

    print(f"Category score: {passed_category}/{total}")
    print(f"Rule-based summary score: {passed_summary}/{total}")


if __name__ == "__main__":
    asyncio.run(run_eval())