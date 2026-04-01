import json
from pathlib import Path

import pytest

from app.agents.rag.runtime import run_langgraph_agent
from app.services.rag_dataset_builder import build_records_from_documents
from app.services.rag_answer_service import NO_ANSWER


def load_benchmark_dataset_v2() -> dict:
    dataset_path = Path("data/rag_benchmark_dataset_v2.json")
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def load_benchmark_cases_v2() -> list[dict]:
    return load_benchmark_dataset_v2()["cases"]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", load_benchmark_cases_v2(), ids=lambda c: c["name"])
async def test_rag_benchmark_dataset_v2(case: dict) -> None:
    dataset = load_benchmark_dataset_v2()
    records = build_records_from_documents(dataset["documents"])

    result = await run_langgraph_agent(
        question=case["question"],
        records=records,
        top_k=3,
        min_score=0.35,
    )

    route = result.get("route")
    answer = result.get("answer", "").lower()
    top_chunks = result.get("top_chunks", [])

    assert route == case["expected_route"], (
        f"Question: {case['question']}\n"
        f"Expected route: {case['expected_route']}\n"
        f"Actual route: {route}\n"
        f"Answer: {result.get('answer', '')}"
    )

    if case["expected_route"] == "answer":
        assert top_chunks, (
            f"Expected retrieved chunks for question: {case['question']}\n"
            f"But top_chunks is empty."
        )

        if "expected_answer_contains" in case:
            for expected in case["expected_answer_contains"]:
                assert expected in answer, (
                    f"Question: {case['question']}\n"
                    f"Expected answer to contain: {expected}\n"
                    f"Actual answer: {result.get('answer', '')}"
                )

        if "expected_answer_contains_any" in case:
            assert any(expected in answer for expected in case["expected_answer_contains_any"]), (
                f"Question: {case['question']}\n"
                f"Expected answer to contain one of: {case['expected_answer_contains_any']}\n"
                f"Actual answer: {result.get('answer', '')}"
            )

    if case["expected_route"] == "fallback":
        assert answer == NO_ANSWER.lower(), (
            f"Question: {case['question']}\n"
            f"Expected fallback answer: {NO_ANSWER}\n"
            f"Actual answer: {result.get('answer', '')}"
        )


@pytest.mark.asyncio
async def test_rag_benchmark_dataset_v2_average_answer_score() -> None:
    dataset = load_benchmark_dataset_v2()
    records = build_records_from_documents(dataset["documents"])
    cases = [
        case
        for case in dataset["cases"]
        if case["expected_route"] == "answer"
    ]

    scores: list[float] = []

    for case in cases:
        result = await run_langgraph_agent(
            question=case["question"],
            records=records,
            top_k=3,
            min_score=0.35,
        )

        top_chunks = result.get("top_chunks", [])
        if top_chunks:
            scores.append(top_chunks[0]["score"])

    assert scores, "No retrieval scores collected for answerable benchmark cases."

    average_score = sum(scores) / len(scores)

    assert average_score >= 0.40, (
        f"Average top retrieval score is too low: {average_score:.4f}"
    )