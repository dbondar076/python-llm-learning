def hit_at_k(top_chunks: list[dict], relevant_doc_ids: list[str]) -> float:
    if not relevant_doc_ids:
        return 0.0

    retrieved_doc_ids = [chunk.get("doc_id") for chunk in top_chunks]
    return 1.0 if any(doc_id in relevant_doc_ids for doc_id in retrieved_doc_ids) else 0.0


def reciprocal_rank(top_chunks: list[dict], relevant_doc_ids: list[str]) -> float:
    if not relevant_doc_ids:
        return 0.0

    for idx, chunk in enumerate(top_chunks, start=1):
        if chunk.get("doc_id") in relevant_doc_ids:
            return 1.0 / idx

    return 0.0


def compute_retrieval_metrics(top_chunks: list[dict], relevant_doc_ids: list[str]) -> dict:
    return {
        "hit_at_k": hit_at_k(top_chunks, relevant_doc_ids),
        "reciprocal_rank": reciprocal_rank(top_chunks, relevant_doc_ids),
    }