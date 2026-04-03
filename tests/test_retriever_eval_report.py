import pytest

from app.services.benchmark_chunked_records_service import build_chunked_records_from_documents
from app.services.rag_eval_service import compute_retrieval_metrics, summarize_metric
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


def build_issue_entry(
    case: dict,
    category: str,
    relevant_doc_ids: list[str],
    top_chunks: list[dict],
    hit: float,
    rr: float,
) -> dict:
    top1_doc_id = top_chunks[0].get("doc_id") if top_chunks else None
    top1_score = float(top_chunks[0].get("score", 0.0)) if top_chunks else 0.0
    returned_doc_ids = [chunk.get("doc_id") for chunk in top_chunks]
    returned_scores = [round(float(chunk.get("score", 0.0)), 4) for chunk in top_chunks]

    return {
        "name": case["name"],
        "category": category,
        "question": case["question"],
        "relevant_doc_ids": relevant_doc_ids,
        "returned_doc_ids": returned_doc_ids,
        "returned_scores": returned_scores,
        "top1_doc_id": top1_doc_id,
        "top1_score": top1_score,
        "hit": hit,
        "rr": rr,
    }


def print_issue_group(title: str, cases: list[dict]) -> None:
    if not cases:
        print(f"\n=== {title} ===")
        print("none")
        return

    print(f"\n=== {title} ===")
    for case in cases:
        print(
            f"- {case['name']} | "
            f"category={case['category']} | "
            f"hit@3={case['hit']:.2f} | "
            f"rr={case['rr']:.2f} | "
            f"top1_doc={case['top1_doc_id']} | "
            f"top1_score={case['top1_score']:.4f}"
        )
        print(f"  question: {case['question']}")
        print(f"  relevant_doc_ids: {case['relevant_doc_ids']}")
        print(f"  returned_doc_ids: {case['returned_doc_ids']}")
        print(f"  returned_scores: {case['returned_scores']}")


@pytest.mark.debug
def test_print_retriever_eval_report() -> None:
    dataset = load_benchmark_dataset_v3()
    records = build_chunked_records_from_documents(dataset["documents"])

    answerable_cases = [
        case
        for case in dataset["cases"]
        if is_answerable_case(case)
    ]

    assert answerable_cases, "No answerable cases found in dataset."

    global_hit_scores: list[float] = []
    global_rr_scores: list[float] = []
    global_top1_scores: list[float] = []

    metrics_by_category: dict[str, dict[str, list[float]]] = {}

    hit_miss_cases: list[dict] = []
    top1_not_relevant_cases: list[dict] = []
    non_top1_rr_cases: list[dict] = []

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
        top1_score = float(top_chunks[0].get("score", 0.0))
        top1_doc_id = top_chunks[0].get("doc_id")
        category = case.get("category", "uncategorized")

        global_hit_scores.append(hit)
        global_rr_scores.append(rr)
        global_top1_scores.append(top1_score)

        if category not in metrics_by_category:
            metrics_by_category[category] = {
                "hit_at_k": [],
                "reciprocal_rank": [],
                "top1_score": [],
            }

        metrics_by_category[category]["hit_at_k"].append(hit)
        metrics_by_category[category]["reciprocal_rank"].append(rr)
        metrics_by_category[category]["top1_score"].append(top1_score)

        print(
            f"{case['name']}: "
            f"category={category}, "
            f"hit@3={hit:.2f}, "
            f"rr={rr:.2f}, "
            f"top1_doc={top1_doc_id}, "
            f"top1_score={top1_score:.4f}"
        )

        issue_entry = build_issue_entry(
            case=case,
            category=category,
            relevant_doc_ids=relevant_doc_ids,
            top_chunks=top_chunks,
            hit=hit,
            rr=rr,
        )

        if hit == 0.0:
            hit_miss_cases.append(issue_entry)

        if top1_doc_id not in relevant_doc_ids:
            top1_not_relevant_cases.append(issue_entry)

        if rr < 1.0:
            non_top1_rr_cases.append(issue_entry)

    print("\n=== Global retrieval summary ===")
    print(f"hit@3 -> {summarize_metric(global_hit_scores)}")
    print(f"mrr    -> {summarize_metric(global_rr_scores)}")
    print(f"top1   -> {summarize_metric(global_top1_scores)}")

    print("\n=== Per-category retrieval summary ===")
    for category in sorted(metrics_by_category):
        category_metrics = metrics_by_category[category]
        print(f"\n[{category}]")
        print(f"hit@3 -> {summarize_metric(category_metrics['hit_at_k'])}")
        print(f"mrr    -> {summarize_metric(category_metrics['reciprocal_rank'])}")
        print(f"top1   -> {summarize_metric(category_metrics['top1_score'])}")

    hit_miss_cases.sort(key=lambda item: (item["rr"], item["top1_score"]))
    top1_not_relevant_cases.sort(key=lambda item: (item["rr"], item["top1_score"]))
    non_top1_rr_cases.sort(key=lambda item: (item["rr"], item["top1_score"]))

    print_issue_group("Cases with hit@3 == 0", hit_miss_cases[:10])
    print_issue_group("Cases where top1 doc is not relevant", top1_not_relevant_cases[:10])
    print_issue_group("Cases with rr < 1.0", non_top1_rr_cases[:10])

    average_hit = sum(global_hit_scores) / len(global_hit_scores)
    average_rr = sum(global_rr_scores) / len(global_rr_scores)
    average_top1 = sum(global_top1_scores) / len(global_top1_scores)

    assert average_hit >= 0.90, f"Average hit@3 too low: {average_hit:.4f}"
    assert average_rr >= 0.85, f"Average MRR too low: {average_rr:.4f}"
    assert average_top1 >= 0.45, f"Average top1 score too low: {average_top1:.4f}"