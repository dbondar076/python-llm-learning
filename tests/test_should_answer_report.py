import pytest

from app.services.benchmark_chunked_records_service import build_chunked_records_from_documents
from app.services.rag_retrieval_service import (
    retrieve_top_chunks_multi_query,
    compute_retrieval_confidence,
    should_answer,
)
from tests.test_rag_benchmark_dataset_v3_chunked import load_benchmark_dataset_v3


TOP_K = 3
MIN_SCORE = 0.35


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


def is_answerable(query: str, chunks: list[dict]) -> bool:
    text = " ".join(chunk["text"].lower() for chunk in chunks[:2])
    q = query.lower()

    # супер простые правила
    if "rating" in q and "rating" not in text:
        return False

    if "budget" in q and "budget" not in text:
        return False

    if "directed" in q and "director" not in text:
        return False

    if "wrote" in q and "author" not in text:
        return False

    return True


def is_answer_case(case: dict) -> bool:
    return case.get("expected_route") == "answer"


def is_fallback_case(case: dict) -> bool:
    return case.get("expected_route") == "fallback"


def test_should_answer_regression_on_answerable_cases() -> None:
    dataset = load_benchmark_dataset_v3()
    records = build_chunked_records_from_documents(dataset["documents"])

    answer_cases = [case for case in dataset["cases"] if is_answer_case(case)]
    assert answer_cases, "No answerable cases found in dataset."

    failed_cases: list[str] = []
    confidences: list[float] = []

    for case in answer_cases:
        top_chunks = retrieve_top_chunks_for_eval(
            question=case["question"],
            records=records,
            top_k=TOP_K,
        )

        assert top_chunks, f"No chunks returned for question: {case['question']}"

        confidence = compute_retrieval_confidence(
            query=case["question"],
            chunks=top_chunks,
        )
        decision = should_answer(
            query=case["question"],
            chunks=top_chunks,
            min_score=MIN_SCORE,
        )

        confidences.append(confidence)

        if not decision:
            failed_cases.append(
                (
                    f"{case['name']} | "
                    f"question={case['question']} | "
                    f"confidence={confidence:.4f} | "
                    f"top1_doc={top_chunks[0].get('doc_id')} | "
                    f"top1_score={float(top_chunks[0].get('score', 0.0)):.4f} | "
                    f"returned_doc_ids={[chunk.get('doc_id') for chunk in top_chunks]}"
                )
            )

    if failed_cases:
        pytest.fail(
            "Some answerable cases were incorrectly routed to fallback:\n"
            + "\n".join(failed_cases)
        )

    average_confidence = sum(confidences) / len(confidences)
    assert average_confidence >= 0.55, (
        f"Average confidence on answerable cases is too low: {average_confidence:.4f}"
    )


def test_should_answer_regression_on_fallback_cases() -> None:
    dataset = load_benchmark_dataset_v3()
    records = build_chunked_records_from_documents(dataset["documents"])

    fallback_cases = [case for case in dataset["cases"] if is_fallback_case(case)]
    assert fallback_cases, "No fallback cases found in dataset."

    wrongly_answered_cases: list[str] = []
    confidences: list[float] = []

    for case in fallback_cases:
        top_chunks = retrieve_top_chunks_for_eval(
            question=case["question"],
            records=records,
            top_k=TOP_K,
        )

        assert top_chunks, f"No chunks returned for question: {case['question']}"

        confidence = compute_retrieval_confidence(
            query=case["question"],
            chunks=top_chunks,
        )
        decision = should_answer(
            query=case["question"],
            chunks=top_chunks,
            min_score=MIN_SCORE,
        )

        confidences.append(confidence)

        if decision:
            wrongly_answered_cases.append(
                (
                    f"{case['name']} | "
                    f"question={case['question']} | "
                    f"confidence={confidence:.4f} | "
                    f"top1_doc={top_chunks[0].get('doc_id')} | "
                    f"top1_score={float(top_chunks[0].get('score', 0.0)):.4f} | "
                    f"returned_doc_ids={[chunk.get('doc_id') for chunk in top_chunks]}"
                )
            )

    if wrongly_answered_cases:
        pytest.fail(
            "Some fallback cases were incorrectly routed to answer:\n"
            + "\n".join(wrongly_answered_cases)
        )

    average_confidence = sum(confidences) / len(confidences)
    assert average_confidence <= 0.60, (
        f"Average confidence on fallback cases is suspiciously high: {average_confidence:.4f}"
    )