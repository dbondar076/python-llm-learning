import json
import re
from typing import Callable, Literal
from pydantic import BaseModel, Field, ValidationError

from app.services.embeddings.service import get_embeddings


class Document(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)


class Chunk(BaseModel):
    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


def load_documents(file_path: str) -> list[Document] | None:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)

        if not isinstance(raw_data, list):
            print("Expected a list of documents")
            return None

        documents: list[Document] = []

        for item in raw_data:
            if not isinstance(item, dict):
                print(f"Invalid document item type: {item}")
                continue

            try:
                document = Document(**item)
                documents.append(document)
            except ValidationError as e:
                print(f"Invalid document: {item}")
                print(e.errors())

        return documents

    except FileNotFoundError:
        print("File not found")
        return None
    except json.JSONDecodeError:
        print("Invalid JSON")
        return None


def split_text_into_chunks(
    text: str,
    chunk_size: int = 80,
    overlap: int = 0
) -> list[str]:
    words = text.split()

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words).strip()

        if chunk_text:
            chunks.append(chunk_text)

        if i + chunk_size >= len(words):
            break

    return chunks


def split_text_into_sentence_chunks(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence for sentence in sentences if sentence]


def build_chunks(
    documents: list[Document],
    chunk_size: int = 12,
    overlap: int = 0,
    strategy: Literal["words", "sentences"] = "words",

) -> list[Chunk]:
    result: list[Chunk] = []

    if strategy == "words":
        split_fn: Callable[[str], list[str]] = lambda text: split_text_into_chunks(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    else:
        split_fn = split_text_into_sentence_chunks

    for doc in documents:
        text_chunks = split_fn(doc.text)

        for index, chunk_text in enumerate(text_chunks, start=1):
            chunk = Chunk(
                doc_id=doc.id,
                title=doc.title,
                chunk_id=f"{doc.id}_chunk_{index}",
                text=chunk_text,
            )
            result.append(chunk)

    return result


def prepare_chunks_with_embeddings(
    file_path: str,
    chunk_size: int = 12,
    overlap: int = 3,
    strategy: Literal["words", "sentences"] = "words",
) -> tuple[list[Chunk], list[list[float]]] | None:
    documents = load_documents(file_path)

    if documents is None:
        return None

    chunks = build_chunks(
        documents,
        chunk_size=chunk_size,
        overlap=overlap,
        strategy=strategy,
    )

    if not chunks:
        return chunks, []

    embeddings = get_embeddings([chunk.text for chunk in chunks])

    return chunks, embeddings