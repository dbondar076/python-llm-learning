import json
import math
from typing import TypedDict


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


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def retrieve_top_chunks_from_index(
    query_embedding: list[float],
    records: list[ChunkEmbeddingRecord],
    top_k: int = 3,
    title_filter: str | None = None,
    doc_id_filter: str | None = None,
) -> list[ScoredChunk]:
    filtered_records = records

    if title_filter is not None:
        filtered_records = [
            record for record in filtered_records
            if record["title"] == title_filter
        ]

    if doc_id_filter is not None:
        filtered_records = [
            record for record in filtered_records
            if record["doc_id"] == doc_id_filter
        ]

    scored: list[ScoredChunk] = []

    for record in filtered_records:
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


if __name__ == "__main__":
    records = load_chunk_embeddings("../app/data/rag/v1/chunk_embeddings.json")

    # сюда пока временно вставь query embedding из предыдущего кода
    # позже мы снова подключим API вызов
    from openai import OpenAI
    from app.settings import OPENAI_API_KEY

    client = OpenAI(api_key=OPENAI_API_KEY)

    question = "API framework in Python"

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question,
    )
    query_embedding = response.data[0].embedding

    print("QUESTION:", question)
    print("=" * 60)

    print("NO FILTER:")
    for item in retrieve_top_chunks_from_index(query_embedding, records):
        print(f"{round(item['score'], 4)} | [{item['title']}] {item['text']}")

    print("\nTITLE FILTER = FastAPI:")
    for item in retrieve_top_chunks_from_index(
        query_embedding,
        records,
        title_filter="FastAPI",
    ):
        print(f"{round(item['score'], 4)} | [{item['title']}] {item['text']}")

    print("\nDOC FILTER = doc1:")
    for item in retrieve_top_chunks_from_index(
        query_embedding,
        records,
        doc_id_filter="doc1",
    ):
        print(f"{round(item['score'], 4)} | [{item['title']}] {item['text']}")