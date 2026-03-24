import asyncio
import time

from app.services.analyzer import analyze_many_async


async def main() -> None:
    texts = [
        "What is Python?",
        "Hi",
        "Python is a programming language.",
        "How do LLMs work?",
        "Hello",
        "What is FastAPI?",
        "Async is useful.",
        "Why use pydantic?",
        "OK",
        "Python works well for APIs.",
    ]

    start = time.perf_counter()
    result = await analyze_many_async(texts)
    elapsed = time.perf_counter() - start

    print(result)
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())