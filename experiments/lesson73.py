import json
import math
from typing import TypedDict

from openai import OpenAI

from app.settings import OPENAI_API_KEY


client = OpenAI(api_key=OPENAI_API_KEY)


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


if __name__ == "__main__":
    records = load_chunk_embeddings("../app/data/rag/v1/chunk_embeddings.json")

    questions = [
        "What can Python be used for?",
        "API framework in Python",
        "What is JavaScript used for?",
    ]

    for question in questions:
        results = retrieve_top_chunks_from_index(question, records)

        print("QUESTION:", question)
        print("RESULTS:")
        for item in results:
            print(f"{round(item['score'], 4)} | [{item['title']}] {item['text']}")
        print("=" * 60)