import math
import re

from app.services.rag_index_service import ChunkEmbeddingRecord
from app.services.openai_client import get_openai_client, reset_openai_client


STOPWORDS = {
    "what",
    "can",
    "be",
    "for",
    "is",
    "a",
    "an",
    "the",
    "it",
    "and",
    "of",
    "in",
    "used",
    "about",
    "tell",
    "me",
    "framework",
    "language",
    "ai",
    "system",
    "tool",
}


class ScoredChunk(ChunkEmbeddingRecord):
    score: float


_query_embedding_cache: dict[str, list[float]] = {}


def reset_runtime_state() -> None:
    _query_embedding_cache.clear()
    reset_openai_client()


def get_query_embedding(text: str) -> list[float]:
    if text in _query_embedding_cache:
        return _query_embedding_cache[text]

    client = get_openai_client()

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )

    embedding = response.data[0].embedding
    _query_embedding_cache[text] = embedding

    return embedding


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def retrieve_top_chunks(
    query: str,
    records: list[ChunkEmbeddingRecord],
    top_k: int = 3,
    title_filter: str | None = None,
    doc_id_filter: str | None = None,
) -> list[ScoredChunk]:
    query_embedding = get_query_embedding(query)

    filtered_records = records

    if title_filter is not None:
        filtered_records = [
            record for record in filtered_records
            if record["title"] == title_filter
        ]

    if doc_id_filter is not None:
        filtered_records = [
            record for record in filtered_records
            if record["doc_id"] == doc_id_filter
        ]

    scored: list[ScoredChunk] = []

    for record in filtered_records:
        score = cosine_similarity(query_embedding, record["embedding"])
        scored.append(
            {
                "doc_id": record["doc_id"],
                "title": record["title"],
                "chunk_id": record["chunk_id"],
                "text": record["text"],
                "embedding": record["embedding"],
                "score": score,
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def has_meaningful_overlap(query: str, chunks: list[ScoredChunk]) -> bool:
    if not chunks:
        return False

    query_tokens = tokenize(query)
    if not query_tokens:
        return False

    top_chunk = chunks[0]
    text_tokens = tokenize(top_chunk["text"])
    title_tokens = tokenize(top_chunk["title"])

    overlap = query_tokens & (text_tokens | title_tokens)
    return len(overlap) > 0


def compute_retrieval_confidence(
    query: str,
    chunks: list[ScoredChunk],
) -> float:
    if not chunks:
        return 0.0

    top_chunk = chunks[0]
    top_score = float(top_chunk["score"])

    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    text_tokens = tokenize(top_chunk["text"])
    title_tokens = tokenize(top_chunk["title"])

    text_overlap = query_tokens & text_tokens
    title_overlap = query_tokens & title_tokens

    overlap_bonus = 0.0

    if text_overlap:
        overlap_bonus += 0.15

    if title_overlap:
        overlap_bonus += 0.10

    confidence = top_score + overlap_bonus
    return min(confidence, 1.0)


def should_answer(
    query: str,
    chunks: list[ScoredChunk],
    min_score: float = 0.52,
) -> bool:
    if not chunks:
        return False

    confidence = compute_retrieval_confidence(query, chunks)

    if confidence >= 0.55:
        return True

    if confidence >= min_score and has_meaningful_overlap(query, chunks):
        return True

    return False


def rerank_chunks(
    query: str,
    chunks: list[ScoredChunk],
) -> list[ScoredChunk]:
    query_tokens = tokenize(query)

    def rerank_score(chunk: ScoredChunk) -> float:
        text_tokens = tokenize(chunk["text"])
        title_tokens = tokenize(chunk["title"])

        text_overlap = len(query_tokens & text_tokens)
        title_overlap = len(query_tokens & title_tokens)

        return chunk["score"] + text_overlap * 0.1 + title_overlap * 0.05

    return sorted(chunks, key=rerank_score, reverse=True)


def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    return {token for token in tokens if token not in STOPWORDS}