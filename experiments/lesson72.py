import json
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


class ChunkEmbeddingRecord(TypedDict):
    doc_id: str
    title: str
    chunk_id: str
    text: str
    embedding: list[float]


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


def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


if __name__ == "__main__":
    documents = load_documents("../documents.json")
    chunks = build_chunks(documents)

    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = get_embeddings(chunk_texts)

    records: list[ChunkEmbeddingRecord] = []

    for chunk, embedding in zip(chunks, embeddings):
        records.append(
            {
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "embedding": embedding,
            }
        )

    with open("../app/data/rag/v1/chunk_embeddings.json", "w", encoding="utf-8") as file:
        json.dump(records, file)

    print(f"Saved {len(records)} chunk embeddings to chunk_embeddings.json")