import json
from pathlib import Path

import pytest

from app.agents.rag.runtime import run_langgraph_agent
from app.services.rag_dataset_builder import build_records_from_documents


DATASET_PATH = Path("data/rag_benchmark_dataset_v1.json")


def load_benchmark_dataset() -> dict:
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_benchmark_cases() -> list[dict]:
    dataset = load_benchmark_dataset()
    return dataset["questions"]


def answer_contains_all(answer: str, expected_values: list[str]) -> bool:
    normalized = answer.lower()
    return all(value.lower() in normalized for value in expected_values)


def answer_contains_any(answer: str, expected_values: list[str]) -> bool:
    normalized = answer.lower()
    return any(value.lower() in normalized for value in expected_values)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", load_benchmark_cases(), ids=lambda c: c["name"])
async def test_rag_benchmark_dataset_v1(case: dict) -> None:
    dataset = load_benchmark_dataset()
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
            f"Question should be answerable but no chunks were returned.\n"
            f"Question: {case['question']}"
        )

        if "expected_answer_contains" in case:
            assert answer_contains_all(answer, case["expected_answer_contains"]), (
                f"Question: {case['question']}\n"
                f"Expected answer to contain all: {case['expected_answer_contains']}\n"
                f"Actual answer: {result.get('answer', '')}"
            )

        if "expected_answer_contains_any" in case:
            assert answer_contains_any(answer, case["expected_answer_contains_any"]), (
                f"Question: {case['question']}\n"
                f"Expected answer to contain any of: {case['expected_answer_contains_any']}\n"
                f"Actual answer: {result.get('answer', '')}"
            )

    if case["expected_route"] == "fallback":
        assert "i don't know" in answer or "do not know" in answer or "no information" in answer, (
            f"Question: {case['question']}\n"
            f"Expected fallback-style answer.\n"
            f"Actual answer: {result.get('answer', '')}"
        )