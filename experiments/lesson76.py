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


class RetrievalCase(TypedDict):
    question: str
    expected_chunk_id: str


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

    cases: list[RetrievalCase] = [
        {
            "question": "What can Python be used for?",
            "expected_chunk_id": "doc1_chunk_2",
        },
        {
            "question": "API framework in Python",
            "expected_chunk_id": "doc2_chunk_1",
        },
        {
            "question": "What provides interactive API documentation?",
            "expected_chunk_id": "doc2_chunk_2",
        },
        {
            "question": "What do large language models work with?",
            "expected_chunk_id": "doc3_chunk_1",
        },
    ]

    top1_hits = 0
    top3_hits = 0

    for case in cases:
        question = case["question"]
        expected_chunk_id = case["expected_chunk_id"]

        results = retrieve_top_chunks_from_index(question, records, top_k=3)
        returned_chunk_ids = [item["chunk_id"] for item in results]

        top1_pass = len(returned_chunk_ids) > 0 and returned_chunk_ids[0] == expected_chunk_id
        top3_pass = expected_chunk_id in returned_chunk_ids

        if top1_pass:
            top1_hits += 1

        if top3_pass:
            top3_hits += 1

        print("QUESTION:", question)
        print("EXPECTED:", expected_chunk_id)
        print("RETURNED:", returned_chunk_ids)
        print("TOP-1 PASS:", top1_pass)
        print("TOP-3 PASS:", top3_pass)
        print("-" * 60)

    total = len(cases)

    print("FINAL RESULTS")
    print(f"Top-1: {top1_hits}/{total}")
    print(f"Top-3: {top3_hits}/{total}")