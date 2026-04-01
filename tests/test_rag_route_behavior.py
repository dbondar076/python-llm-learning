import pytest

from app.services.benchmark_chunked_records_service import build_chunked_records_from_documents
from app.agents.rag.runtime import run_langgraph_agent
from tests.test_rag_benchmark_dataset_v3_chunked import load_benchmark_dataset_v3

NO_ANSWER = "I don't know based on the provided context."


@pytest.mark.asyncio
@pytest.mark.parametrize("case", load_benchmark_dataset_v3()["cases"], ids=lambda c: c["name"])
async def test_route_matches_final_answer_semantics(case: dict) -> None:
    dataset = load_benchmark_dataset_v3()
    records = build_chunked_records_from_documents(dataset["documents"])

    result = await run_langgraph_agent(
        question=case["question"],
        records=records,
        top_k=3,
        min_score=0.35,
    )

    route = result.get("route")
    answer = (result.get("answer") or "").strip()
    top_chunks = result.get("top_chunks", [])

    if answer == NO_ANSWER:
        assert route == "fallback", (
            f"Answer is NO_ANSWER but route is not fallback.\n"
            f"Question: {case['question']}\n"
            f"Route: {route}\n"
        )

    if route == "answer":
        assert answer and answer != NO_ANSWER, (
            f"Route is answer but answer is empty or NO_ANSWER.\n"
            f"Question: {case['question']}\n"
            f"Answer: {answer}\n"
        )
        assert top_chunks, (
            f"Route is answer but no top_chunks were returned.\n"
            f"Question: {case['question']}\n"
        )