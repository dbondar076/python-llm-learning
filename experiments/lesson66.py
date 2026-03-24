import json
import math
import re
from collections import Counter
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


class ScoredChunkInt(TypedDict):
    doc_id: str
    title: str
    chunk_id: str
    text: str
    score: int


class ScoredChunkFloat(TypedDict):
    doc_id: str
    title: str
    chunk_id: str
    text: str
    score: float


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


def tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    return [token for token in raw_tokens if token not in STOPWORDS]


# ----------------------------
# KEYWORD
# ----------------------------

def score_chunk_keyword(query: str, chunk: Chunk) -> int:
    query_tokens = set(tokenize(query))
    chunk_tokens = set(tokenize(chunk["text"]))
    title_tokens = set(tokenize(chunk["title"]))

    score = 0

    for token in query_tokens:
        if token in title_tokens:
            score += 3
        if token in chunk_tokens:
            score += 1

    return score


def retrieve_top_chunks_keyword(query: str, chunks: list[Chunk], top_k: int = 3) -> list[ScoredChunkInt]:
    scored: list[ScoredChunkInt] = []

    for chunk in chunks:
        score = score_chunk_keyword(query, chunk)
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

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ----------------------------
# VECTOR
# ----------------------------

def build_vocabulary(texts: list[str]) -> list[str]:
    vocab: set[str] = set()

    for text in texts:
        vocab.update(tokenize(text))

    return sorted(vocab)


def text_to_vector(text: str, vocabulary: list[str]) -> list[float]:
    counts = Counter(tokenize(text))
    return [float(counts[word]) for word in vocabulary]


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def retrieve_top_chunks_vector(
    query: str,
    chunks: list[Chunk],
    vocabulary: list[str],
    top_k: int = 3,
) -> list[ScoredChunkFloat]:
    query_vector = text_to_vector(query, vocabulary)
    scored: list[ScoredChunkFloat] = []

    for chunk in chunks:
        chunk_vector = text_to_vector(chunk["text"], vocabulary)
        score = cosine_similarity(query_vector, chunk_vector)

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

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ----------------------------
# TF-IDF
# ----------------------------

def compute_idf(texts: list[str], vocabulary: list[str]) -> dict[str, float]:
    total_docs = len(texts)
    tokenized_texts = [set(tokenize(text)) for text in texts]
    idf: dict[str, float] = {}

    for word in vocabulary:
        docs_with_word = sum(1 for tokens in tokenized_texts if word in tokens)
        idf[word] = math.log((total_docs + 1) / (docs_with_word + 1)) + 1

    return idf


def text_to_tfidf_vector(text: str, vocabulary: list[str], idf: dict[str, float]) -> list[float]:
    tokens = tokenize(text)
    counts = Counter(tokens)
    total_tokens = len(tokens)

    if total_tokens == 0:
        return [0.0 for _ in vocabulary]

    vector: list[float] = []

    for word in vocabulary:
        tf = counts[word] / total_tokens
        vector.append(tf * idf[word])

    return vector


def retrieve_top_chunks_tfidf(
    query: str,
    chunks: list[Chunk],
    vocabulary: list[str],
    idf: dict[str, float],
    top_k: int = 3,
) -> list[ScoredChunkFloat]:
    query_vector = text_to_tfidf_vector(query, vocabulary, idf)
    scored: list[ScoredChunkFloat] = []

    for chunk in chunks:
        chunk_vector = text_to_tfidf_vector(chunk["text"], vocabulary, idf)
        score = cosine_similarity(query_vector, chunk_vector)

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

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    documents = load_documents("../documents.json")
    chunks = build_chunks(documents)

    all_texts = [chunk["text"] for chunk in chunks]
    vocabulary = build_vocabulary(all_texts)
    idf = compute_idf(all_texts, vocabulary)

    questions = [
        "What can Python be used for?",
        "What is JavaScript used for?",
        "API framework in Python",
    ]

    for question in questions:
        print("QUESTION:", question)
        print("=" * 60)

        print("KEYWORD:")
        for item in retrieve_top_chunks_keyword(question, chunks):
            print(f"{item['score']} | {item['text']}")

        print("\nVECTOR:")
        for item in retrieve_top_chunks_vector(question, chunks, vocabulary):
            print(f"{round(item['score'], 3)} | {item['text']}")

        print("\nTF-IDF:")
        for item in retrieve_top_chunks_tfidf(question, chunks, vocabulary, idf):
            print(f"{round(item['score'], 3)} | {item['text']}")

        print("\n" + "#" * 60 + "\n")