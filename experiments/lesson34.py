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


async def run_eval() -> None:
    passed = 0
    total = len(test_cases)

    for case in test_cases:
        result = await analyze_with_llm_async(case["input"])
        ok = result.category == case["expected_category"]

        print("INPUT:", case["input"])
        print("EXPECTED:", case["expected_category"])
        print("ACTUAL:", result.category)
        print("PASS:", ok)
        print("-" * 40)

        if ok:
            passed += 1

    print(f"Final score: {passed}/{total}")


if __name__ == "__main__":
    asyncio.run(run_eval())