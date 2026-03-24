import asyncio

from app.services.analyzer import analyze_many_safe_async


async def main() -> None:
    texts = [
        "What is Python?",
        "RETRY: What is Python?",
        "TIMEOUT: What is Python?",
        "Hi",
    ]

    results = await analyze_many_safe_async(texts)

    for item in results:
        print(item)


if __name__ == "__main__":
    asyncio.run(main())