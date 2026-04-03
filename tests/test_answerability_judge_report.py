import pytest

from app.services.benchmark_chunked_records_service import build_chunked_records_from_documents
from app.services.rag_judge_service import judge_retrieval_answerability
from app.services.rag_retrieval_service import retrieve_top_chunks_multi_query
from tests.test_rag_benchmark_dataset_v3_chunked import load_benchmark_dataset_v3


TOP_K = 3


def retrieve_top_chunks_for_eval(
    question: str,
    records: list[dict],
    top_k: int = TOP_K,
) -> list[dict]:
    return retrieve_top_chunks_multi_query(
        query=question,
        records=records,
        top_k=top_k,
        per_query_k=5,
    )


def is_answer_case(case: dict) -> bool:
    return case.get("expected_route") == "answer"


def is_fallback_case(case: dict) -> bool:
    return case.get("expected_route") == "fallback"


@pytest.mark.asyncio
async def test_answerability_judge_on_answerable_cases() -> None:
    dataset = load_benchmark_dataset_v3()
    records = build_chunked_records_from_documents(dataset["documents"])

    answer_cases = [case for case in dataset["cases"] if is_answer_case(case)]
    assert answer_cases, "No answerable cases found in dataset."

    failed_cases: list[str] = []

    for case in answer_cases:
        top_chunks = retrieve_top_chunks_for_eval(
            question=case["question"],
            records=records,
            top_k=TOP_K,
        )

        assert top_chunks, f"No chunks returned for question: {case['question']}"

        decision = await judge_retrieval_answerability(
            question=case["question"],
            chunks=top_chunks,
        )

        if not decision:
            failed_cases.append(
                (
                    f"{case['name']} | "
                    f"question={case['question']} | "
                    f"top1_doc={top_chunks[0].get('doc_id')} | "
                    f"top1_score={float(top_chunks[0].get('score', 0.0)):.4f}"
                )
            )

    if failed_cases:
        pytest.fail(
            "Some answerable cases were rejected by answerability judge:\n"
            + "\n".join(failed_cases)
        )


@pytest.mark.asyncio
async def test_answerability_judge_on_fallback_cases() -> None:
    dataset = load_benchmark_dataset_v3()
    records = build_chunked_records_from_documents(dataset["documents"])

    fallback_cases = [case for case in dataset["cases"] if is_fallback_case(case)]
    assert fallback_cases, "No fallback cases found in dataset."

    wrong_cases: list[str] = []

    for case in fallback_cases:
        top_chunks = retrieve_top_chunks_for_eval(
            question=case["question"],
            records=records,
            top_k=TOP_K,
        )

        assert top_chunks, f"No chunks returned for question: {case['question']}"

        decision = await judge_retrieval_answerability(
            question=case["question"],
            chunks=top_chunks,
        )

        if decision:
            wrong_cases.append(
                (
                    f"{case['name']} | "
                    f"question={case['question']} | "
                    f"top1_doc={top_chunks[0].get('doc_id')} | "
                    f"top1_score={float(top_chunks[0].get('score', 0.0)):.4f} | "
                    f"returned_doc_ids={[chunk.get('doc_id') for chunk in top_chunks]}"
                )
            )

    if wrong_cases:
        pytest.fail(
            "Some fallback cases were incorrectly approved by answerability judge:\n"
            + "\n".join(wrong_cases)
        )