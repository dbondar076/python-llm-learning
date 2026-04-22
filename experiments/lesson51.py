import asyncio
import re
from pydantic import BaseModel, Field

from app.services.llm_service import run_text_prompt_with_retry_async
from app.services.offline.preprocessing import prepare_chunks, Chunk


class ScoredChunk(BaseModel):
    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    score: int = Field(ge=1)


STOP_WORDS = {
    "a", "an", "the", "is", "are", "am", "be", "can", "for", "of", "to",
    "what", "how", "why", "when", "where", "in", "on", "at", "and", "or"
}
NO_ANSWER = "I don't know based on the provided context."


def tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return [token for token in raw_tokens if token not in STOP_WORDS]


def score_chunk_tokens(query_tokens: set[str], chunk: Chunk) -> int:
    chunk_tokens = set(tokenize(chunk.text))
    return len(query_tokens & chunk_tokens)


def retrieve_top_chunks(
    query: str,
    chunks: list[Chunk],
    top_k: int = 3,
) -> list[ScoredChunk]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    query_tokens = set(tokenize(query))
    scored_chunks: list[ScoredChunk] = []

    for chunk in chunks:
        chunk_score = score_chunk_tokens(query_tokens, chunk)

        if chunk_score > 0:
            scored_chunk = ScoredChunk(
                doc_id=chunk.doc_id,
                title=chunk.title,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=chunk_score,
            )
            scored_chunks.append(scored_chunk)

    scored_chunks.sort(key=lambda item: item.score, reverse=True)
    return scored_chunks[:top_k]


def build_context(chunks: list[ScoredChunk]) -> str:
    parts: list[str] = []

    for chunk in chunks:
        parts.append(f"[{chunk.title} | {chunk.chunk_id} | score={chunk.score}]\n{chunk.text}")

    return "\n\n".join(parts)


def build_rag_prompt(question: str, context: str) -> str:
    return (
        "Answer the user's question using only the provided context.\n"
        "If the answer is not in the context, say exactly: I don't know based on the provided context.\n"
        "Be concise.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )


async def answer_with_rag(question: str, chunks: list[Chunk]) -> tuple[str, list[ScoredChunk]]:
    top_chunks = retrieve_top_chunks(question, chunks, top_k=3)

    if not top_chunks:
        return NO_ANSWER, []

    context = build_context(top_chunks)
    prompt = build_rag_prompt(question, context)
    answer = await run_text_prompt_with_retry_async(prompt)

    return answer.strip(), top_chunks


async def main() -> None:
    chunks = prepare_chunks("../documents.json", chunk_size=12, overlap=3)

    if chunks is None:
        print("Failed to prepare chunks.")
        return

    question = "What can Python be used for?"
    answer, top_chunks = await answer_with_rag(question, chunks)

    print("QUESTION:", question)
    print()

    print("RETRIEVED CHUNKS:")
    for item in top_chunks:
        print(item)
    print()

    print("ANSWER:")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())