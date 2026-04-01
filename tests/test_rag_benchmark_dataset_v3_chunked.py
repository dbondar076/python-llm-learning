import json
from pathlib import Path

import pytest

from app.agents.rag.runtime import run_langgraph_agent
from app.services.benchmark_chunked_records_service import build_chunked_records_from_documents


DATASET_PATH = Path("data/rag_benchmark_dataset_v3.json")


def load_benchmark_dataset_v3() -> dict:
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_benchmark_cases_v3() -> list[dict]:
    dataset = load_benchmark_dataset_v3()
    return dataset["cases"]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", load_benchmark_cases_v3(), ids=lambda c: c["name"])
async def test_rag_benchmark_dataset_v3_chunked(case: dict) -> None:
    dataset = load_benchmark_dataset_v3()
    records = build_chunked_records_from_documents(dataset["documents"])

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

    if route == "answer":
        assert top_chunks, (
            f"Question: {case['question']}\n"
            "Expected retrieved chunks for answerable case, but got none."
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

    if route == "fallback":
        assert "don't know" in answer or "do not know" in answer, (
            f"Question: {case['question']}\n"
            "Expected fallback-style answer.\n"
            f"Actual answer: {result.get('answer', '')}"
        )


@pytest.mark.asyncio
async def test_rag_benchmark_dataset_v3_chunked_average_answer_score() -> None:
    dataset = load_benchmark_dataset_v3()
    records = build_chunked_records_from_documents(dataset["documents"])
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