import re
from typing import TypedDict

from app.settings import RAG_MIN_SCORE, RAG_TOP_K
from app.services.llm_service import run_text_prompt_with_retry_async
from app.services.rag_index_service import ChunkEmbeddingRecord
from app.services.rag_retrieval_service import (
    ScoredChunk,
    should_answer,
    retrieve_top_chunks_with_rerank,
)

NO_ANSWER = "I don't know based on the provided context."


class MergedChunk(TypedDict):
    doc_id: str
    title: str
    chunk_ids: list[str]
    text: str
    score: float


def parse_chunk_number(chunk_id: str) -> int | None:
    match = re.search(r"_chunk_(\d+)$", chunk_id)
    if not match:
        return None

    return int(match.group(1))


def merge_adjacent_chunks(chunks: list[ScoredChunk]) -> list[MergedChunk]:
    if not chunks:
        return []

    indexed_chunks = [
        {
            **chunk,
            "_original_rank": i,
        }
        for i, chunk in enumerate(chunks)
    ]

    sorted_chunks = sorted(
        indexed_chunks,
        key=lambda c: (
            c["doc_id"],
            parse_chunk_number(c["chunk_id"]) or 0,
        ),
    )

    merged: list[dict] = []
    current: dict | None = None

    for chunk in sorted_chunks:
        chunk_num = parse_chunk_number(chunk["chunk_id"])

        if current is None:
            current = {
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "chunk_ids": [chunk["chunk_id"]],
                "text": chunk["text"],
                "score": chunk["score"],
                "last_chunk_num": chunk_num,
                "_best_rank": chunk["_original_rank"],
            }
            continue

        same_doc = current["doc_id"] == chunk["doc_id"]
        adjacent = (
            current["last_chunk_num"] is not None
            and chunk_num is not None
            and chunk_num == current["last_chunk_num"] + 1
        )

        if same_doc and adjacent:
            current["chunk_ids"].append(chunk["chunk_id"])
            current["text"] += " " + chunk["text"]
            current["score"] = max(current["score"], chunk["score"])
            current["last_chunk_num"] = chunk_num
            current["_best_rank"] = min(current["_best_rank"], chunk["_original_rank"])
        else:
            merged.append(current)
            current = {
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "chunk_ids": [chunk["chunk_id"]],
                "text": chunk["text"],
                "score": chunk["score"],
                "last_chunk_num": chunk_num,
                "_best_rank": chunk["_original_rank"],
            }

    if current is not None:
        merged.append(current)

    merged.sort(key=lambda c: c["_best_rank"])

    result: list[MergedChunk] = []
    for chunk in merged:
        result.append(
            {
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "chunk_ids": chunk["chunk_ids"],
                "text": chunk["text"],
                "score": chunk["score"],
            }
        )

    return result


def build_context(chunks: list[MergedChunk]) -> str:
    return "\n".join(
        f"[{chunk['title']} | {', '.join(chunk['chunk_ids'])}] {chunk['text']}"
        for chunk in chunks
    )


def build_rag_prompt(question: str, context: str) -> str:
    return (
        "Answer the user's question using only the provided context.\n"
        "If the answer is not in the context, say exactly: I don't know based on the provided context.\n"
        "Do not guess.\n"
        "Be concise.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )


async def answer_with_rag(
    question: str,
    records: list[ChunkEmbeddingRecord],
    top_k: int = RAG_TOP_K,
    min_score: float = RAG_MIN_SCORE,
    title_filter: str | None = None,
    doc_id_filter: str | None = None,
) -> tuple[list[ScoredChunk], str]:
    top_chunks = retrieve_top_chunks_with_rerank(
        query=question,
        records=records,
        top_k=top_k,
        title_filter=title_filter,
        doc_id_filter=doc_id_filter,
        initial_k=max(10, top_k),
    )

    if not should_answer(question, top_chunks, min_score=min_score):
        return top_chunks, NO_ANSWER

    merged_chunks = merge_adjacent_chunks(top_chunks)
    context = build_context(merged_chunks)
    prompt = build_rag_prompt(question, context)
    answer = await run_text_prompt_with_retry_async(prompt)

    return top_chunks, answer