import math
import re
from collections import Counter


texts = [
    "Python is used for AI and automation",
    "FastAPI is a Python framework for APIs",
    "Large language models work with text",
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


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


if __name__ == "__main__":
    vocabulary = build_vocabulary(texts)

    query = "Python for APIs"
    query_vector = text_to_vector(query, vocabulary)

    print("VOCABULARY:")
    print(vocabulary)
    print()

    print("QUERY:", query)
    print("QUERY VECTOR:")
    print(query_vector)
    print()

    for text in texts:
        vector = text_to_vector(text, vocabulary)
        similarity = cosine_similarity(query_vector, vector)

        print("TEXT:", text)
        print("VECTOR:", vector)
        print("SIMILARITY:", round(similarity, 3))
        print("-" * 50)