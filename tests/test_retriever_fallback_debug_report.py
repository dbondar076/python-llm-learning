import pytest

from app.services.benchmark_chunked_records_service import build_chunked_records_from_documents
from app.services.rag_eval_service import summarize_metric
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


def is_fallback_case(case: dict) -> bool:
    return case.get("expected_route") == "fallback"


@pytest.mark.debug
def test_print_fallback_retriever_report() -> None:
    dataset = load_benchmark_dataset_v3()
    records = build_chunked_records_from_documents(dataset["documents"])

    fallback_cases = [
        case
        for case in dataset["cases"]
        if is_fallback_case(case)
    ]

    assert fallback_cases, "No fallback cases found in dataset."

    top1_scores: list[float] = []
    metrics_by_category: dict[str, list[float]] = {}

    high_score_cases: list[dict] = []
    very_high_score_cases: list[dict] = []

    for case in fallback_cases:
        top_chunks = retrieve_top_chunks_for_eval(
            question=case["question"],
            records=records,
            top_k=3,
        )

        assert top_chunks, f"No chunks returned for question: {case['question']}"

        category = case.get("category", "uncategorized")
        top1_doc_id = top_chunks[0].get("doc_id")
        top1_score = float(top_chunks[0].get("score", 0.0))
        returned_doc_ids = [chunk.get("doc_id") for chunk in top_chunks]

        top1_scores.append(top1_score)
        metrics_by_category.setdefault(category, []).append(top1_score)

        print(
            f"{case['name']}: "
            f"category={category}, "
            f"top1_doc={top1_doc_id}, "
            f"top1_score={top1_score:.4f}, "
            f"returned_doc_ids={returned_doc_ids}"
        )

        if top1_score >= 0.60:
            high_score_cases.append(
                {
                    "name": case["name"],
                    "category": category,
                    "question": case["question"],
                    "top1_doc_id": top1_doc_id,
                    "top1_score": top1_score,
                    "returned_doc_ids": returned_doc_ids,
                }
            )

        if top1_score >= 0.75:
            very_high_score_cases.append(
                {
                    "name": case["name"],
                    "category": category,
                    "question": case["question"],
                    "top1_doc_id": top1_doc_id,
                    "top1_score": top1_score,
                    "returned_doc_ids": returned_doc_ids,
                }
            )

    print("\n=== Global fallback retrieval summary ===")
    print(f"top1 -> {summarize_metric(top1_scores)}")

    print("\n=== Per-category fallback retrieval summary ===")
    for category in sorted(metrics_by_category):
        print(f"\n[{category}]")
        print(f"top1 -> {summarize_metric(metrics_by_category[category])}")

    if high_score_cases:
        high_score_cases.sort(key=lambda item: item["top1_score"], reverse=True)
        print("\n=== High-score fallback cases (>= 0.60) ===")
        for case in high_score_cases:
            print(
                f"- {case['name']} | "
                f"category={case['category']} | "
                f"top1_doc={case['top1_doc_id']} | "
                f"top1_score={case['top1_score']:.4f}"
            )
            print(f"  question: {case['question']}")
            print(f"  returned_doc_ids: {case['returned_doc_ids']}")

    if very_high_score_cases:
        very_high_score_cases.sort(key=lambda item: item["top1_score"], reverse=True)
        print("\n=== Very high-score fallback cases (>= 0.75) ===")
        for case in very_high_score_cases:
            print(
                f"- {case['name']} | "
                f"category={case['category']} | "
                f"top1_doc={case['top1_doc_id']} | "
                f"top1_score={case['top1_score']:.4f}"
            )
            print(f"  question: {case['question']}")
            print(f"  returned_doc_ids: {case['returned_doc_ids']}")