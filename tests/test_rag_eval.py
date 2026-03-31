import json
from pathlib import Path

import pytest

from app.agents.rag.runtime import run_langgraph_agent
from app.services.rag_index_service import load_chunk_embeddings


def load_eval_dataset() -> list[dict]:
    dataset_path = Path("tests/fixtures/rag_eval_dataset.json")
    with dataset_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", load_eval_dataset())
async def test_rag_eval_dataset(case: dict) -> None:
    records = load_chunk_embeddings()

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

    print(
        f"\nQ: {case['question']}\n"
        f"Score: {score:.2f}\n"
        f"Answer: {result['answer']}\n"
    )

    assert matches >= len(expected_keywords) // 2, (
        f"Question: {case['question']}\n"
        f"Answer: {result['answer']}\n"
        f"Expected one of: {expected_keywords}"
    )


@pytest.mark.asyncio
async def test_rag_eval_score():
    records = load_chunk_embeddings()
    dataset = load_eval_dataset()

    total_score = 0

    for case in dataset:
        result = await run_langgraph_agent(
            question=case["question"],
            records=records,
        )

        answer = result["answer"].lower()
        expected_keywords = [kw.lower() for kw in case["expected_keywords"]]

        matches = sum(1 for kw in expected_keywords if kw in answer)
        score = matches / len(expected_keywords)

        total_score += score

    avg_score = total_score / len(dataset)

    print(f"\nAVG SCORE: {avg_score:.2f}")

    assert avg_score >= 0.5