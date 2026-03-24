import asyncio
import time


async def mock_llm_call(text: str) -> str:
    await asyncio.sleep(1)
    return f"Processed: {text}"


async def process_sequential(texts: list[str]) -> list[str]:
    results: list[str] = []

    for text in texts:
        result = await mock_llm_call(text)
        results.append(result)

    return results


async def process_concurrent(texts: list[str]) -> list[str]:
    tasks = [mock_llm_call(text) for text in texts]
    return await asyncio.gather(*tasks)


async def main() -> None:
    texts = ["one", "two", "three"]

    start = time.perf_counter()
    sequential_result = await process_sequential(texts)
    sequential_time = time.perf_counter() - start

    start = time.perf_counter()
    concurrent_result = await process_concurrent(texts)
    concurrent_time = time.perf_counter() - start

    print("Sequential:", sequential_result)
    print(f"Sequential time: {sequential_time:.2f}s")

    print("Concurrent:", concurrent_result)
    print(f"Concurrent time: {concurrent_time:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())