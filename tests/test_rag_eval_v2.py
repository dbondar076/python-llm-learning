import json
from pathlib import Path

import pytest

from app.agents.rag.runtime import run_langgraph_agent
from app.services.rag_index_service import load_chunk_embeddings


def load_eval_dataset_v2() -> list[dict]:
    dataset_path = Path("tests/fixtures/rag_eval_dataset_v2.json")
    with dataset_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", load_eval_dataset_v2(), ids=lambda c: c["name"])
async def test_rag_eval_dataset_v2(case: dict) -> None:
    records = load_chunk_embeddings()

    result = await run_langgraph_agent(
        question=case["question"],
        records=records,
        top_k=3,
        min_score=0.35,
    )

    answer = result["answer"].lower()
    route = result.get("route")
    expected_route = case["expected_route"]

    assert route == expected_route, (
        f"Question: {case['question']}\n"
        f"Expected route: {expected_route}\n"
        f"Actual route: {route}\n"
        f"Answer: {result['answer']}"
    )

    expected_keywords = [kw.lower() for kw in case.get("expected_keywords", [])]

    if expected_route == "answer":
        matches = sum(1 for kw in expected_keywords if kw in answer)
        min_required = max(1, len(expected_keywords) // 2)

        assert matches >= min_required, (
            f"Question: {case['question']}\n"
            f"Route: {route}\n"
            f"Answer: {result['answer']}\n"
            f"Expected keywords: {expected_keywords}\n"
            f"Matches: {matches}/{len(expected_keywords)}"
        )

    if expected_route == "fallback":
        forbidden_phrases = [
            "rust is",
            "moonscript ultra ai framework is",
        ]

        for phrase in forbidden_phrases:
            assert phrase not in answer, (
                f"Question: {case['question']}\n"
                f"Route: {route}\n"
                f"Answer should not confidently invent facts.\n"
                f"Found forbidden phrase: {phrase}\n"
                f"Full answer: {result['answer']}"
            )


@pytest.mark.asyncio
async def test_rag_eval_v2_average_answer_score() -> None:
    records = load_chunk_embeddings()
    dataset = load_eval_dataset_v2()

    answerable_cases = [c for c in dataset if c["expected_route"] == "answer"]
    total_score = 0.0

    for case in answerable_cases:
        result = await run_langgraph_agent(
            question=case["question"],
            records=records,
            top_k=3,
            min_score=0.35,
        )

        answer = result["answer"].lower()
        expected_keywords = [kw.lower() for kw in case["expected_keywords"]]

        matches = sum(1 for kw in expected_keywords if kw in answer)
        score = matches / len(expected_keywords)
        total_score += score

    avg_score = total_score / len(answerable_cases)

    print(f"\nAVG ANSWERABLE SCORE: {avg_score:.2f}")

    assert avg_score >= 0.5