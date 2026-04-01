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


def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return {token for token in tokens if token not in STOPWORDS}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _extract_ordinal_hints(query: str) -> set[str]:
    query_norm = normalize_text(query)
    hints: set[str] = set()

    mapping = {
        "first": {"first", "1st"},
        "second": {"second", "2nd"},
        "third": {"third", "3rd"},
        "fourth": {"fourth", "4th"},
        "fifth": {"fifth", "5th"},
        "last": {"last", "final", "fifth", "5th"},
    }

    for trigger, values in mapping.items():
        if trigger in query_norm:
            hints.update(values)

    return hints


def _extract_phrase_hints(query: str) -> set[str]:
    query_norm = normalize_text(query)
    phrases: set[str] = set()

    known_phrases = {
        "solar drift",
        "glass harbor",
        "moonline echo",
        "silent orbit",
        "ember falls",
        "favorite books",
        "travel notes",
        "watchlist",
        "personal watchlist",
    }

    for phrase in known_phrases:
        if phrase in query_norm:
            phrases.add(phrase)

    return phrases


def _lexical_overlap_ratio(query_tokens: set[str], chunk_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0

    overlap = query_tokens & chunk_tokens
    return len(overlap) / len(query_tokens)


def _compute_rerank_score(query: str, chunk: ScoredChunk) -> float:
    """
    Поверх cosine score добавляем легкие lexical / hint boosts.
    Важно:
    - не ломаем исходный score
    - не меняем структуру ScoredChunk
    - бусты ограниченные, чтобы не переопределять retrieval полностью
    """
    base_score = float(chunk["score"])

    query_norm = normalize_text(query)
    chunk_text_norm = normalize_text(chunk["text"])
    chunk_title_norm = normalize_text(chunk["title"])
    combined_norm = f"{chunk_title_norm} {chunk_text_norm}"

    query_tokens = tokenize(query)
    chunk_text_tokens = tokenize(chunk["text"])
    chunk_title_tokens = tokenize(chunk["title"])
    combined_tokens = chunk_text_tokens | chunk_title_tokens

    score = base_score

    # 1. Обычный lexical overlap
    overlap_ratio = _lexical_overlap_ratio(query_tokens, combined_tokens)
    score += overlap_ratio * 0.20

    # 2. Phrase boosts
    phrase_hints = _extract_phrase_hints(query_norm)
    for phrase in phrase_hints:
        if phrase in combined_norm:
            score += 0.12

    # 3. Ordinal boosts (first / third / last ...)
    ordinal_hints = _extract_ordinal_hints(query_norm)
    has_matching_ordinal = False

    for hint in ordinal_hints:
        if hint in combined_norm:
            score += 0.18
            has_matching_ordinal = True

    # Если вопрос про ordinal, а chunk явно про другой ordinal — небольшой штраф
    if ordinal_hints and not has_matching_ordinal:
        known_ordinals = {
            "first", "1st",
            "second", "2nd",
            "third", "3rd",
            "fourth", "4th",
            "fifth", "5th",
            "last", "final",
        }
        if any(token in combined_norm for token in known_ordinals):
            score -= 0.07

    # 4. Специальный boost для doc-like hints
    if "watchlist" in query_norm and "watchlist" in combined_norm:
        score += 0.05

    if "favorite books" in query_norm and (
        "favorite books" in combined_norm or "books" in combined_norm
    ):
        score += 0.05

    if "travel notes" in query_norm and (
        "travel notes" in combined_norm or "travel" in combined_norm
    ):
        score += 0.05

    return score


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


def retrieve_top_chunks_with_rerank(
    query: str,
    records: list[ChunkEmbeddingRecord],
    top_k: int = 3,
    initial_k: int = 10,
    title_filter: str | None = None,
    doc_id_filter: str | None = None,
) -> list[ScoredChunk]:
    """
    Безопасная надстройка над retrieve_top_chunks.
    Сначала dense retrieval, потом локальный rerank top-N.
    """
    initial_chunks = retrieve_top_chunks(
        query=query,
        records=records,
        top_k=max(top_k, initial_k),
        title_filter=title_filter,
        doc_id_filter=doc_id_filter,
    )

    return rerank_chunks(query=query, chunks=initial_chunks)[:top_k]


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


def should_answer_with_override(
    query: str,
    chunks: list[ScoredChunk],
) -> bool:
    if not chunks:
        return False

    confidence = compute_retrieval_confidence(query, chunks)

    # базовое правило (оставляем как есть)
    if confidence >= 0.55:
        return True

    # новое правило: если запрос "явный" + есть хоть какой-то сигнал
    if confidence >= 0.45 and is_high_intent_query(query):
        return True

    # fallback: overlap
    if confidence >= 0.52 and has_meaningful_overlap(query, chunks):
        return True

    return False


def rerank_chunks(
    query: str,
    chunks: list[ScoredChunk],
) -> list[ScoredChunk]:
    """
    Возвращает тот же list[ScoredChunk], только пересортированный.
    ВАЖНО:
    - тип данных не меняем
    - ключи не добавляем
    - score перезаписываем rerank-значением, чтобы downstream логика
      (`should_answer`, route decision) работала уже на улучшенном top chunk
    """
    reranked: list[ScoredChunk] = []

    for chunk in chunks:
        reranked_chunk: ScoredChunk = {
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "embedding": chunk["embedding"],
            "score": _compute_rerank_score(query, chunk),
        }
        reranked.append(reranked_chunk)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked


def is_high_intent_query(query: str) -> bool:
    """
    Явные вопросы, где почти всегда нужно отвечать, а не clarify.
    """
    q = query.lower()

    triggers = [
        "who",
        "what",
        "which",
        "name",
        "list",
        "first",
        "second",
        "third",
        "last",
        "director",
        "actor",
    ]

    return any(t in q for t in triggers)