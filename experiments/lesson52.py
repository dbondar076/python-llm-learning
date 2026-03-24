import json
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


def load_documents(file_path: str) -> list[Document]:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def split_text_into_chunks(
    text: str,
    chunk_size: int = 12,
    overlap: int = 4,
) -> list[str]:
    words = text.split()
    chunks: list[str] = []

    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk_size must be greater than overlap")

    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if not chunk_words:
            continue

        chunks.append(" ".join(chunk_words))

        if i + chunk_size >= len(words):
            break

    return chunks


def build_chunks(documents: list[Document]) -> list[Chunk]:
    result: list[Chunk] = []

    for doc in documents:
        text_chunks = split_text_into_chunks(doc["text"], chunk_size=12, overlap=4)

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


if __name__ == "__main__":
    documents = load_documents("../documents.json")
    chunks = build_chunks(documents)

    for chunk in chunks:
        print(chunk)