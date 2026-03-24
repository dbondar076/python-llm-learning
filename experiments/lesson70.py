import asyncio
import json
import math
import re
from typing import TypedDict

from openai import OpenAI

from app.settings import OPENAI_API_KEY
from app.services.llm_service import run_text_prompt_with_retry_async


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


NO_ANSWER = "I don't know based on the provided context."


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


def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


def retrieve_top_chunks_semantic(
    query: str,
    chunks: list[Chunk],
    chunk_embeddings: list[list[float]],
    top_k: int = 3,
) -> list[ScoredChunk]:
    query_embedding = get_embeddings([query])[0]
    scored: list[ScoredChunk] = []

    for chunk, chunk_embedding in zip(chunks, chunk_embeddings):
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


def build_context(chunks: list[ScoredChunk]) -> str:
    return "\n".join(
        f"[{chunk['title']} | {chunk['chunk_id']}] {chunk['text']}"
        for chunk in chunks
    )


def build_rag_prompt(question: str, context: str) -> str:
    return (
        "Answer the user's question using only the provided context.\n"
        "If the answer is not in the context, say exactly: I don't know based on the provided context.\n"
        "Do not guess.\n"
        "Be concise.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )


def should_answer_with_context(chunks: list[ScoredChunk], min_score: float = 0.52) -> bool:
    if not chunks:
        return False

    return chunks[0]["score"] >= min_score


async def answer_with_semantic_rag(
    question: str,
    chunks: list[Chunk],
    chunk_embeddings: list[list[float]],
) -> str:
    top_chunks = retrieve_top_chunks_semantic(
        question,
        chunks,
        chunk_embeddings,
        top_k=3,
    )

    print("RETRIEVED CHUNKS:")
    for chunk in top_chunks:
        print(f"{round(chunk['score'], 4)} | [{chunk['title']}] {chunk['text']}")
    print("-" * 50)

    if not should_answer_with_context(top_chunks, min_score=0.52):
        return NO_ANSWER

    context = build_context(top_chunks)
    prompt = build_rag_prompt(question, context)
    return await run_text_prompt_with_retry_async(prompt)


async def main() -> None:
    documents = load_documents("../documents.json")
    chunks = build_chunks(documents)

    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = get_embeddings(chunk_texts)

    questions = [
        "What can Python be used for?",
        "API framework in Python",
        "What is JavaScript used for?",
    ]

    for question in questions:
        answer = await answer_with_semantic_rag(question, chunks, chunk_embeddings)

        print("QUESTION:", question)
        print("ANSWER:", answer)
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())