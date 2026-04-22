import asyncio

from pydantic import BaseModel, Field

from app.services.llm_service import run_text_prompt_with_retry_async
from app.services.offline.preprocessing import prepare_chunks_with_embeddings, Chunk
from app.services.embeddings.service import get_embeddings
from app.services.embeddings.utils import cosine_similarity


class ScoredChunk(BaseModel):
    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    score: float


NO_ANSWER = "I don't know based on the provided context."


def retrieve_top_chunks_semantic(
    query: str,
    chunks: list[Chunk],
    chunk_embeddings: list[list[float]],
    top_k: int = 3,
) -> list[ScoredChunk]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if len(chunks) != len(chunk_embeddings):
        raise ValueError("chunks and chunk_embeddings must have the same length")

    query_embedding = get_embeddings([query])[0]
    scored: list[ScoredChunk] = []

    for chunk, chunk_embedding in zip(chunks, chunk_embeddings):
        score = cosine_similarity(query_embedding, chunk_embedding)

        scored_chunk = ScoredChunk(
            doc_id=chunk.doc_id,
            title=chunk.title,
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            score=score,
        )
        scored.append(scored_chunk)

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]


def build_context(chunks: list[ScoredChunk]) -> str:
    return "\n\n".join(
        f"[{chunk.title} | {chunk.chunk_id} | score={round(chunk.score, 4)}]\n{chunk.text}"
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


def should_answer_with_context(
    chunks: list[ScoredChunk],
    min_score: float = 0.52
) -> bool:
    if not chunks:
        return False

    return chunks[0].score >= min_score


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
        print(f"{round(chunk.score, 4)} | [{chunk.title}] {chunk.text}")
    print("-" * 50)

    if not should_answer_with_context(top_chunks, min_score=0.52):
        return NO_ANSWER

    context = build_context(top_chunks)
    prompt = build_rag_prompt(question, context)
    return (await run_text_prompt_with_retry_async(prompt)).strip()


async def main() -> None:
    prepared_data = prepare_chunks_with_embeddings(
        "../documents.json",
        chunk_size=12,
        overlap=3,
        strategy="sentences"
    )

    if prepared_data is None:
        print("Failed to prepare chunks.")
        return

    chunks, chunk_embeddings = prepared_data

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