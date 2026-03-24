import asyncio
import json
import math
from typing import TypedDict

from openai import OpenAI

from app.settings import OPENAI_API_KEY
from app.services.llm_service import run_text_prompt_with_retry_async


client = OpenAI(api_key=OPENAI_API_KEY)

NO_ANSWER = "I don't know based on the provided context."


class ChunkEmbeddingRecord(TypedDict):
    doc_id: str
    title: str
    chunk_id: str
    text: str
    embedding: list[float]


class ScoredChunk(TypedDict):
    doc_id: str
    title: str
    chunk_id: str
    text: str
    score: float


class RagEvalCase(TypedDict):
    question: str
    expected_chunk_id: str
    expected_answer: str


def load_chunk_embeddings(file_path: str) -> list[ChunkEmbeddingRecord]:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def retrieve_top_chunks_from_index(
    query: str,
    records: list[ChunkEmbeddingRecord],
    top_k: int = 3,
) -> list[ScoredChunk]:
    query_embedding = get_embedding(query)
    scored: list[ScoredChunk] = []

    for record in records:
        score = cosine_similarity(query_embedding, record["embedding"])
        scored.append(
            {
                "doc_id": record["doc_id"],
                "title": record["title"],
                "chunk_id": record["chunk_id"],
                "text": record["text"],
                "score": score,
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def should_answer(chunks: list[ScoredChunk], min_score: float = 0.52) -> bool:
    if not chunks:
        return False

    return chunks[0]["score"] >= min_score


def build_context(chunks: list[ScoredChunk]) -> str:
    return "\n".join(
        f"[{chunk['title']} | {chunk['chunk_id']}] {chunk['text']}"
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


async def answer_with_rag_from_index(
    question: str,
    records: list[ChunkEmbeddingRecord],
) -> tuple[list[ScoredChunk], str]:
    top_chunks = retrieve_top_chunks_from_index(question, records, top_k=3)

    if not should_answer(top_chunks, min_score=0.52):
        return top_chunks, NO_ANSWER

    context = build_context(top_chunks)
    prompt = build_rag_prompt(question, context)
    answer = await run_text_prompt_with_retry_async(prompt)
    return top_chunks, answer


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


async def main() -> None:
    records = load_chunk_embeddings("../app/data/rag/v1/chunk_embeddings.json")

    cases: list[RagEvalCase] = [
        {
            "question": "What can Python be used for?",
            "expected_chunk_id": "doc1_chunk_2",
            "expected_answer": "Python can be used for web development, automation, data analysis, and AI.",
        },
        {
            "question": "API framework in Python",
            "expected_chunk_id": "doc2_chunk_1",
            "expected_answer": "FastAPI is a modern Python framework for building APIs.",
        },
        {
            "question": "What provides interactive API documentation?",
            "expected_chunk_id": "doc2_chunk_2",
            "expected_answer": "Interactive API documentation is provided by FastAPI.",
        },
        {
            "question": "What do large language models work with?",
            "expected_chunk_id": "doc3_chunk_1",
            "expected_answer": "Large language models work with text tokens.",
        },
    ]

    top1_hits = 0
    top3_hits = 0
    answer_hits = 0

    for case in cases:
        question = case["question"]
        expected_chunk_id = case["expected_chunk_id"]
        expected_answer = case["expected_answer"]

        top_chunks, actual_answer = await answer_with_rag_from_index(question, records)
        returned_chunk_ids = [item["chunk_id"] for item in top_chunks]

        top1_pass = len(returned_chunk_ids) > 0 and returned_chunk_ids[0] == expected_chunk_id
        top3_pass = expected_chunk_id in returned_chunk_ids
        answer_pass = normalize_text(expected_answer) == normalize_text(actual_answer)

        if top1_pass:
            top1_hits += 1

        if top3_pass:
            top3_hits += 1

        if answer_pass:
            answer_hits += 1

        print("QUESTION:", question)
        print("EXPECTED CHUNK:", expected_chunk_id)
        print("RETURNED CHUNKS:", returned_chunk_ids)
        print("TOP-1 PASS:", top1_pass)
        print("TOP-3 PASS:", top3_pass)
        print("EXPECTED ANSWER:", expected_answer)
        print("ACTUAL ANSWER:", actual_answer)
        print("ANSWER PASS:", answer_pass)
        print("-" * 70)

    total = len(cases)

    print("FINAL RESULTS")
    print(f"Retrieval Top-1: {top1_hits}/{total}")
    print(f"Retrieval Top-3: {top3_hits}/{total}")
    print(f"Answer Accuracy: {answer_hits}/{total}")


if __name__ == "__main__":
    asyncio.run(main())