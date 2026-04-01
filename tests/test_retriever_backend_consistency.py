import pytest

from app.services.benchmark_chunked_records_service import build_chunked_records_from_documents
from tests.test_rag_benchmark_dataset_v3_chunked import load_benchmark_dataset_v3
from app.services.rag_retrieval_service import retrieve_top_chunks_with_rerank


def retrieve_top_chunks_for_test(
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


RETRIEVAL_CASES = [
    {
        "name": "watchlist_first_item",
        "question": "What is the first movie in the personal watchlist?",
        "expected_doc_id": "doc_watchlist_v3",
        "expected_text_contains_any": ["silent orbit"],
        "min_top1_score": 0.40,
    },
    {
        "name": "watchlist_third_item",
        "question": "What is the third movie in the personal watchlist?",
        "expected_doc_id": "doc_watchlist_v3",
        "expected_text_contains_any": ["ember falls"],
        "min_top1_score": 0.40,
    },
    {
        "name": "watchlist_last_item",
        "question": "What is the last movie in the personal watchlist?",
        "expected_doc_id": "doc_watchlist_v3",
        "expected_text_contains_any": ["moonline echo"],
        "min_top1_score": 0.40,
    },
    {
        "name": "director_solar_drift",
        "question": "Who directed the movie Solar Drift?",
        "expected_doc_id": "doc_movies_v3",
        "expected_text_contains_any": ["anna kovacs", "solar drift"],
        "min_top1_score": 0.40,
    },
    {
        "name": "director_glass_harbor",
        "question": "Who directed the movie Glass Harbor?",
        "expected_doc_id": "doc_movies_v3",
        "expected_text_contains_any": ["thomas vale", "glass harbor"],
        "min_top1_score": 0.40,
    },
    {
        "name": "director_moonline_echo",
        "question": "Who directed the movie Moonline Echo?",
        "expected_doc_id": "doc_movies_v3",
        "expected_text_contains_any": ["pavel orlov", "moonline echo"],
        "min_top1_score": 0.40,
    },
    {
        "name": "actor_solar_drift",
        "question": "Who is the lead actor in the movie Solar Drift?",
        "expected_doc_id": "doc_movies_v3",
        "expected_text_contains_any": ["lena hart", "solar drift"],
        "min_top1_score": 0.40,
    },
    {
        "name": "favorite_books",
        "question": "Name one favorite book from the favorite books document.",
        "expected_doc_id": "doc_books_v3",
        "expected_text_contains_any": [
            "dune",
            "foundation",
            "hyperion",
            "the left hand of darkness",
        ],
        "min_top1_score": 0.35,
    },
    {
        "name": "travel_notes",
        "question": "Name one city mentioned in the travel notes document.",
        "expected_doc_id": "doc_travel_v3",
        "expected_text_contains_any": ["lisbon", "kyoto", "reykjavik"],
        "min_top1_score": 0.35,
    },
]


@pytest.mark.parametrize("case", RETRIEVAL_CASES, ids=lambda c: c["name"])
def test_retriever_returns_expected_document(case: dict) -> None:
    dataset = load_benchmark_dataset_v3()
    records = build_chunked_records_from_documents(dataset["documents"])

    top_chunks = retrieve_top_chunks_for_test(
        question=case["question"],
        records=records,
        top_k=3,
    )

    assert top_chunks, f"No chunks returned for question: {case['question']}"

    top_1 = top_chunks[0]
    top_1_doc_id = top_1.get("doc_id")
    top_1_score = top_1.get("score", 0.0)
    top_1_text = (top_1.get("text") or "").lower()

    all_doc_ids = [chunk.get("doc_id") for chunk in top_chunks]

    assert case["expected_doc_id"] in all_doc_ids, (
        f"Expected doc_id '{case['expected_doc_id']}' not found in top-{len(top_chunks)}.\n"
        f"Question: {case['question']}\n"
        f"Returned doc_ids: {all_doc_ids}\n"
        f"Top chunks: {top_chunks}"
    )

    assert top_1_doc_id == case["expected_doc_id"], (
        f"Unexpected top-1 doc_id.\n"
        f"Question: {case['question']}\n"
        f"Expected: {case['expected_doc_id']}\n"
        f"Actual: {top_1_doc_id}\n"
        f"Top chunks: {top_chunks}"
    )

    assert top_1_score >= case["min_top1_score"], (
        f"Top-1 score is too low.\n"
        f"Question: {case['question']}\n"
        f"Expected >= {case['min_top1_score']}\n"
        f"Actual: {top_1_score:.4f}\n"
        f"Top chunk: {top_1}"
    )

    assert any(
        any(token in (chunk.get("text") or "").lower() for token in case["expected_text_contains_any"])
        for chunk in top_chunks
    ), (
        f"Returned chunks do not contain expected signal.\n"
        f"Question: {case['question']}\n"
        f"Expected any of: {case['expected_text_contains_any']}\n"
        f"Top chunks: {top_chunks}"
    )


def test_retriever_average_top1_score_on_answerable_cases() -> None:
    dataset = load_benchmark_dataset_v3()
    records = build_chunked_records_from_documents(dataset["documents"])

    scores: list[float] = []

    for case in RETRIEVAL_CASES:
        top_chunks = retrieve_top_chunks_for_test(
            question=case["question"],
            records=records,
            top_k=3,
        )

        assert top_chunks, f"No chunks returned for question: {case['question']}"
        scores.append(top_chunks[0].get("score", 0.0))

    average_score = sum(scores) / len(scores)

    assert average_score >= 0.45, (
        f"Average top-1 retrieval score is too low: {average_score:.4f}\n"
        f"Scores: {[round(score, 4) for score in scores]}"
    )


def test_debug_single_question() -> None:
    dataset = load_benchmark_dataset_v3()
    records = build_chunked_records_from_documents(dataset["documents"])

    question = "Who directed the movie Solar Drift?"
    top_chunks = retrieve_top_chunks_for_test(question=question, records=records, top_k=1)

    chunk = top_chunks[0]
    print(type(chunk))
    print(chunk)
    print(vars(chunk) if hasattr(chunk, "__dict__") else "no __dict__")