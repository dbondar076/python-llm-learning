import json
import math
import re
from typing import TypedDict

from openai import OpenAI
from app.settings import OPENAI_API_KEY


client = OpenAI(api_key=OPENAI_API_KEY)


class Document(TypedDict):
    id: str
    title: str
    text: str


class Chunk(TypedDict):
    doc_id: str
    title: str
    chunk_id: str
    text: str


class ScoredChunk(TypedDict):
    doc_id: str
    title: str
    chunk_id: str
    text: str
    score: float


def load_documents(file_path: str) -> list[Document]:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def split_text_into_sentence_chunks(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence for sentence in sentences if sentence]


def build_chunks(documents: list[Document]) -> list[Chunk]:
    result: list[Chunk] = []

    for doc in documents:
        sentences = split_text_into_sentence_chunks(doc["text"])

        for i, sentence in enumerate(sentences, start=1):
            result.append(
                {
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "chunk_id": f"{doc['id']}_chunk_{i}",
                    "text": sentence,
                }
            )

    return result


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def retrieve_top_chunks_semantic(
    query: str,
    chunks: list[Chunk],
    top_k: int = 3,
) -> list[ScoredChunk]:
    query_embedding = get_embedding(query)
    scored: list[ScoredChunk] = []

    for chunk in chunks:
        chunk_embedding = get_embedding(chunk["text"])
        score = cosine_similarity(query_embedding, chunk_embedding)

        scored.append(
            {
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": score,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    documents = load_documents("../documents.json")
    chunks = build_chunks(documents)

    questions = [
        "What can Python be used for?",
        "API framework in Python",
        "What is JavaScript used for?",
    ]

    for question in questions:
        results = retrieve_top_chunks_semantic(question, chunks, top_k=3)

        print("QUESTION:", question)
        print("RESULTS:")
        for item in results:
            print(f"{round(item['score'], 4)} | [{item['title']}] {item['text']}")
        print("=" * 60)