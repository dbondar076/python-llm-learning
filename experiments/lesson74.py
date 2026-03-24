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
) -> str:
    top_chunks = retrieve_top_chunks_from_index(question, records, top_k=3)

    print("RETRIEVED CHUNKS:")
    for chunk in top_chunks:
        print(f"{round(chunk['score'], 4)} | [{chunk['title']}] {chunk['text']}")
    print("-" * 50)

    if not should_answer(top_chunks, min_score=0.52):
        return NO_ANSWER

    context = build_context(top_chunks)
    prompt = build_rag_prompt(question, context)
    return await run_text_prompt_with_retry_async(prompt)


async def main() -> None:
    records = load_chunk_embeddings("../app/data/rag/v1/chunk_embeddings.json")

    questions = [
        "What can Python be used for?",
        "API framework in Python",
        "What is JavaScript used for?",
    ]

    for question in questions:
        answer = await answer_with_rag_from_index(question, records)

        print("QUESTION:", question)
        print("ANSWER:", answer)
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())