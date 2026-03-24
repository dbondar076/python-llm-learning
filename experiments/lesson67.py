import math

from openai import OpenAI
from app.settings import OPENAI_API_KEY


client = OpenAI(api_key=OPENAI_API_KEY)


texts = [
    "Python is used for AI and automation",
    "FastAPI is a Python framework for APIs",
    "Large language models work with text",
]

query = "Python for APIs"


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


if __name__ == "__main__":
    query_embedding = get_embedding(query)

    print("QUERY:", query)
    print("EMBEDDING DIM:", len(query_embedding))
    print()

    for text in texts:
        text_embedding = get_embedding(text)
        similarity = cosine_similarity(query_embedding, text_embedding)

        print("TEXT:", text)
        print("SIMILARITY:", round(similarity, 4))
        print("-" * 50)