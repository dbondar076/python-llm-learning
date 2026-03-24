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


STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "what",
    "how",
    "why",
    "can",
    "be",
    "for",
    "of",
    "and",
    "to",
    "it",
    "in",
    "on",
    "with",
}


def load_documents(file_path: str) -> list[Document]:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def split_text_into_word_chunks(
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


def split_text_into_sentence_chunks(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence for sentence in sentences if sentence]


def build_chunks(
    documents: list[Document],
    strategy: str,
) -> list[Chunk]:
    result: list[Chunk] = []

    for doc in documents:
        if strategy == "word":
            text_chunks = split_text_into_word_chunks(doc["text"], chunk_size=12, overlap=4)
        elif strategy == "sentence":
            text_chunks = split_text_into_sentence_chunks(doc["text"])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        for index, chunk_text in enumerate(text_chunks, start=1):
            result.append(
                {
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "chunk_id": f'{doc["id"]}_{strategy}_chunk_{index}',
                    "text": chunk_text,
                }
            )

    return result


def tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    return [token for token in raw_tokens if token not in STOPWORDS]


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


async def answer_with_rag(question: str, chunks: list[Chunk]) -> tuple[str, str]:
    top_chunks = retrieve_top_chunks(question, chunks, top_k=3)
    context = build_context(top_chunks)
    prompt = build_rag_prompt(question, context)
    answer = await run_text_prompt_with_retry_async(prompt)
    return context, answer


async def main() -> None:
    documents = load_documents("../documents.json")
    question = "What can Python be used for?"

    word_chunks = build_chunks(documents, strategy="word")
    sentence_chunks = build_chunks(documents, strategy="sentence")

    word_context, word_answer = await answer_with_rag(question, word_chunks)
    sentence_context, sentence_answer = await answer_with_rag(question, sentence_chunks)

    print("QUESTION:", question)
    print("=" * 60)

    print("\nWORD-OVERLAP CHUNKS")
    print("-" * 30)
    print("CONTEXT:")
    print(word_context)
    print()
    print("ANSWER:")
    print(word_answer)

    print("\n" + "=" * 60)

    print("\nSENTENCE CHUNKS")
    print("-" * 30)
    print("CONTEXT:")
    print(sentence_context)
    print()
    print("ANSWER:")
    print(sentence_answer)


if __name__ == "__main__":
    asyncio.run(main())