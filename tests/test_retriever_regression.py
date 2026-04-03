import pytest

from app.services.benchmark_chunked_records_service import build_chunked_records_from_documents
from app.services.rag_eval_service import compute_retrieval_metrics
from app.services.rag_retrieval_service import retrieve_top_chunks_with_rerank
from tests.test_rag_benchmark_dataset_v3_chunked import load_benchmark_dataset_v3


def retrieve_top_chunks_for_eval(
    question: str,
    records: list[dict],
    top_k: int = 3,
) -> list[dict]:
    return retrieve_top_chunks_with_rerank(
        query=question,
        records=records,
        top_k=top_k,
        initial_k=10,
    )


def is_answerable_case(case: dict) -> bool:
    return case.get("expected_route") == "answer"


def test_retriever_regression_on_answerable_cases() -> None:
    dataset = load_benchmark_dataset_v3()
    records = build_chunked_records_from_documents(dataset["documents"])

    answerable_cases = [
        case
        for case in dataset["cases"]
        if is_answerable_case(case)
    ]

    assert answerable_cases, "No answerable cases found in dataset."

    hit_scores: list[float] = []
    rr_scores: list[float] = []
    top1_scores: list[float] = []

    hit_miss_cases: list[str] = []
    top1_not_relevant_cases: list[str] = []

    for case in answerable_cases:
        relevant_doc_ids = case.get("relevant_doc_ids", [])
        assert relevant_doc_ids, (
            f"Case '{case['name']}' is missing relevant_doc_ids."
        )

        top_chunks = retrieve_top_chunks_for_eval(
            question=case["question"],
            records=records,
            top_k=3,
        )

        assert top_chunks, f"No chunks returned for question: {case['question']}"

        metrics = compute_retrieval_metrics(
            top_chunks=top_chunks,
            relevant_doc_ids=relevant_doc_ids,
        )

        hit = metrics["hit_at_k"]
        rr = metrics["reciprocal_rank"]
        top1_doc_id = top_chunks[0].get("doc_id")
        top1_score = float(top_chunks[0].get("score", 0.0))

        hit_scores.append(hit)
        rr_scores.append(rr)
        top1_scores.append(top1_score)

        if hit == 0.0:
            hit_miss_cases.append(
                (
                    f"{case['name']} | "
                    f"question={case['question']} | "
                    f"relevant_doc_ids={relevant_doc_ids} | "
                    f"returned_doc_ids={[chunk.get('doc_id') for chunk in top_chunks]}"
                )
            )

        if top1_doc_id not in relevant_doc_ids:
            top1_not_relevant_cases.append(
                (
                    f"{case['name']} | "
                    f"question={case['question']} | "
                    f"relevant_doc_ids={relevant_doc_ids} | "
                    f"top1_doc_id={top1_doc_id} | "
                    f"returned_doc_ids={[chunk.get('doc_id') for chunk in top_chunks]}"
                )
            )

    assert not hit_miss_cases, (
        "Retriever missed relevant docs completely for some answerable cases:\n- "
        + "\n- ".join(hit_miss_cases)
    )

    assert not top1_not_relevant_cases, (
        "Retriever returned non-relevant top1 doc for some answerable cases:\n- "
        + "\n- ".join(top1_not_relevant_cases)
    )

    average_hit = sum(hit_scores) / len(hit_scores)
    average_rr = sum(rr_scores) / len(rr_scores)
    average_top1 = sum(top1_scores) / len(top1_scores)

    assert average_hit >= 0.95, (
        f"Average hit@3 too low: {average_hit:.4f}"
    )
    assert average_rr >= 0.90, (
        f"Average MRR too low: {average_rr:.4f}"
    )
    assert average_top1 >= 0.50, (
        f"Average top1 score too low: {average_top1:.4f}"
    )