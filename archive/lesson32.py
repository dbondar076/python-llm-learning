import asyncio
import time

from app.services.llm_service import analyze_with_llm_async


async def main() -> None:
    text = "What is Python?"

    start = time.perf_counter()
    result1 = await analyze_with_llm_async(text)
    elapsed1 = time.perf_counter() - start
    print(result1)
    print(f"First call: {elapsed1:.2f}s")

    start = time.perf_counter()
    result2 = await analyze_with_llm_async(text)
    elapsed2 = time.perf_counter() - start
    print(result2)
    print(f"Second call: {elapsed2:.2f}s")

    print("Sleeping to let cache expire...")
    await asyncio.sleep(4)

    start = time.perf_counter()
    result3 = await analyze_with_llm_async(text)
    elapsed3 = time.perf_counter() - start
    print(result3)
    print(f"Third call after TTL: {elapsed3:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())