import asyncio
import re

from app.services.rag_answer_service import NO_ANSWER, build_rag_prompt
from app.services.rag_index_service import load_chunk_embeddings
from app.services.rag_retrieval_service import retrieve_top_chunks, should_answer
from app.services.llm_service import run_text_prompt_with_retry_async


def parse_chunk_number(chunk_id: str) -> int | None:
    match = re.search(r"_chunk_(\d+)$", chunk_id)
    if not match:
        return None

    return int(match.group(1))


def merge_adjacent_chunks(chunks: list[dict]) -> list[dict]:
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
    current = None

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

    for chunk in merged:
        chunk.pop("_best_rank", None)
        chunk.pop("last_chunk_num", None)

    return merged


def build_merged_context(chunks: list[dict]) -> str:
    return "\n".join(
        f"[{chunk['title']} | {', '.join(chunk['chunk_ids'])}] {chunk['text']}"
        for chunk in chunks
    )


async def answer_with_merged_context(question: str) -> None:
    records = load_chunk_embeddings()
    top_chunks = retrieve_top_chunks(question, records, top_k=3)

    print("RETRIEVED CHUNKS:")
    for chunk in top_chunks:
        print(f"{round(chunk['score'], 4)} | [{chunk['chunk_id']}] {chunk['text']}")
    print("-" * 60)

    merged_chunks = merge_adjacent_chunks(top_chunks)

    print("MERGED CHUNKS:")
    for chunk in merged_chunks:
        print(f"{round(chunk['score'], 4)} | [{', '.join(chunk['chunk_ids'])}] {chunk['text']}")
    print("-" * 60)

    if not should_answer(top_chunks, min_score=0.52):
        print("ANSWER:", NO_ANSWER)
        return

    context = build_merged_context(merged_chunks)
    prompt = build_rag_prompt(question, context)
    answer = await run_text_prompt_with_retry_async(prompt)

    print("QUESTION:", question)
    print("ANSWER:", answer)
    print("=" * 60)


async def main() -> None:
    questions = [
        "What can Python be used for?",
        "API framework in Python",
        "What is JavaScript used for?",
    ]

    for question in questions:
        await answer_with_merged_context(question)


if __name__ == "__main__":
    asyncio.run(main())