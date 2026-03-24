import json
import re
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
    score: int


def load_documents(file_path: str) -> list[Document]:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def split_text_into_chunks(text: str, chunk_size: int = 80) -> list[str]:
    words = text.split()
    chunks: list[str] = []

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))

    return chunks


def build_chunks(documents: list[Document]) -> list[Chunk]:
    result: list[Chunk] = []

    for doc in documents:
        text_chunks = split_text_into_chunks(doc["text"], chunk_size=12)

        for index, chunk_text in enumerate(text_chunks, start=1):
            result.append(
                {
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "chunk_id": f'{doc["id"]}_chunk_{index}',
                    "text": chunk_text,
                }
            )

    return result


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def score_chunk(query: str, chunk: Chunk) -> int:
    query_tokens = set(tokenize(query))
    chunk_tokens = set(tokenize(chunk["text"]))
    return len(query_tokens & chunk_tokens)


def retrieve_top_chunks(query: str, chunks: list[Chunk], top_k: int = 3) -> list[ScoredChunk]:
    scored: list[ScoredChunk] = []

    for chunk in chunks:
        score = score_chunk(query, chunk)

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

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    documents = load_documents("../documents.json")
    chunks = build_chunks(documents)

    query = "What can Python be used for?"
    results = retrieve_top_chunks(query, chunks, top_k=3)

    print("QUERY:", query)
    print()

    for item in results:
        print(item)