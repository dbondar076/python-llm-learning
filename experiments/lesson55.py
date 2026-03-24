import asyncio
import json
import re
from typing import TypedDict

from app.services.llm_service import run_text_prompt_with_retry_async


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


NO_ANSWER = "I don't know based on the provided context."


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


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def score_chunk(query: str, chunk: Chunk) -> int:
    query_tokens = set(tokenize(query))
    chunk_tokens = set(tokenize(chunk["text"]))
    return len(query_tokens & chunk_tokens)


def retrieve_top_chunks(
    query: str,
    chunks: list[Chunk],
    top_k: int = 3,
) -> list[ScoredChunk]:
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


def build_context(chunks: list[ScoredChunk]) -> str:
    parts: list[str] = []

    for chunk in chunks:
        parts.append(f"[{chunk['title']} | {chunk['chunk_id']}] {chunk['text']}")

    return "\n".join(parts)


def build_rag_prompt(question: str, context: str) -> str:
    return (
        "Answer the user's question using only the provided context.\n"
        "If the answer is not in the context, say exactly: I don't know based on the provided context.\n"
        "Do not guess.\n"
        "Be concise.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )


def should_answer_with_context(chunks: list[ScoredChunk], min_score: int = 2) -> bool:
    if not chunks:
        return False

    best_score = chunks[0]["score"]
    return best_score >= min_score


async def answer_with_rag(question: str, chunks: list[Chunk]) -> str:
    top_chunks = retrieve_top_chunks(question, chunks, top_k=3)

    print("RETRIEVED CHUNKS:")
    print(top_chunks)
    print("-" * 50)

    if not should_answer_with_context(top_chunks, min_score=2):
        return NO_ANSWER

    context = build_context(top_chunks)
    prompt = build_rag_prompt(question, context)
    return await run_text_prompt_with_retry_async(prompt)


async def main() -> None:
    documents = load_documents("../documents.json")
    chunks = build_chunks(documents)

    questions = [
        "What can Python be used for?",
        "What is JavaScript used for?",
    ]

    for question in questions:
        answer = await answer_with_rag(question, chunks)

        print("QUESTION:", question)
        print("ANSWER:", answer)
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())