from app.services.benchmark_chunked_records_service import build_chunked_records_from_documents
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


def test_retriever_regression_on_fallback_cases() -> None:
    dataset = load_benchmark_dataset_v3()
    records = build_chunked_records_from_documents(dataset["documents"])

    fallback_cases = [
        case
        for case in dataset["cases"]
        if is_fallback_case(case)
    ]

    assert fallback_cases, "No fallback cases found in dataset."

    top1_scores: list[float] = []
    high_score_cases: list[str] = []
    very_high_score_cases: list[str] = []

    for case in fallback_cases:
        top_chunks = retrieve_top_chunks_for_eval(
            question=case["question"],
            records=records,
            top_k=3,
        )

        assert top_chunks, f"No chunks returned for question: {case['question']}"

        top1_doc_id = top_chunks[0].get("doc_id")
        top1_score = float(top_chunks[0].get("score", 0.0))
        returned_doc_ids = [chunk.get("doc_id") for chunk in top_chunks]

        top1_scores.append(top1_score)

        print(
            f"{case['name']}: "
            f"top1_doc={top1_doc_id}, "
            f"top1_score={top1_score:.4f}, "
            f"returned_doc_ids={returned_doc_ids}"
        )

        if top1_score >= 0.60:
            high_score_cases.append(
                f"{case['name']} | "
                f"question={case['question']} | "
                f"top1_doc={top1_doc_id} | "
                f"top1_score={top1_score:.4f} | "
                f"returned_doc_ids={returned_doc_ids}"
            )

        if top1_score >= 0.75:
            very_high_score_cases.append(
                f"{case['name']} | "
                f"question={case['question']} | "
                f"top1_doc={top1_doc_id} | "
                f"top1_score={top1_score:.4f} | "
                f"returned_doc_ids={returned_doc_ids}"
            )

    average_top1 = sum(top1_scores) / len(top1_scores)
    max_top1 = max(top1_scores)

    print("\n=== Fallback retrieval stats ===")
    print(f"average_top1_score={average_top1:.4f}")
    print(f"max_top1_score={max_top1:.4f}")
    print(f"high_score_cases_count={len(high_score_cases)}")
    print(f"very_high_score_cases_count={len(very_high_score_cases)}")

    if high_score_cases:
        print("\n=== High-score fallback cases (>= 0.60) ===")
        for item in high_score_cases:
            print(f"- {item}")

    if very_high_score_cases:
        print("\n=== Very high-score fallback cases (>= 0.75) ===")
        for item in very_high_score_cases:
            print(f"- {item}")

    assert average_top1 <= 0.62, (
        f"Average fallback top1 score too high: {average_top1:.4f}"
    )

    assert len(very_high_score_cases) <= 2, (
        "Too many very-high-score fallback cases:\n- "
        + "\n- ".join(very_high_score_cases)
    )