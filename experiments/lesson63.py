import json
import math
import re
from collections import Counter
from typing import TypedDict


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
    return re.split(r"(?<=[.!?])\s+", text.strip())


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


# ----------------------------
# VECTOR BUILDING
# ----------------------------

def build_vocabulary(texts: list[str]) -> list[str]:
    vocab: set[str] = set()

    for text in texts:
        vocab.update(tokenize(text))

    return sorted(vocab)


def text_to_vector(text: str, vocabulary: list[str]) -> list[float]:
    counts = Counter(tokenize(text))
    return [float(counts[word]) for word in vocabulary]


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


# ----------------------------
# BUILD CHUNKS
# ----------------------------

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


# ----------------------------
# SEMANTIC RETRIEVAL
# ----------------------------

def retrieve_top_chunks_semantic(
    query: str,
    chunks: list[Chunk],
    vocabulary: list[str],
    top_k: int = 3,
) -> list[ScoredChunk]:
    query_vector = text_to_vector(query, vocabulary)

    scored: list[ScoredChunk] = []

    for chunk in chunks:
        chunk_vector = text_to_vector(chunk["text"], vocabulary)
        score = cosine_similarity(query_vector, chunk_vector)

        if score > 0:
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


# ----------------------------
# MAIN
# ----------------------------

if __name__ == "__main__":
    documents = load_documents("../documents.json")
    chunks = build_chunks(documents)

    # 🔑 важный момент — vocabulary строим по ВСЕМ chunk'ам
    all_texts = [chunk["text"] for chunk in chunks]
    vocabulary = build_vocabulary(all_texts)

    questions = [
        "What can Python be used for?",
        "What is JavaScript used for?",
        "API framework in Python",
    ]

    for question in questions:
        results = retrieve_top_chunks_semantic(
            question,
            chunks,
            vocabulary,
        )

        print("QUESTION:", question)
        print("RESULTS:")
        for r in results:
            print(f"{round(r['score'], 3)} | {r['text']}")
        print("=" * 60)