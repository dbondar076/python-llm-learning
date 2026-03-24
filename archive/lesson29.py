import asyncio

from app.services.llm_service import (
    LLMRetryError,
    LLMTimeoutError,
    analyze_with_llm_async,
)


async def main() -> None:
    print("=== Success case ===")
    result = await analyze_with_llm_async("What is Python?")
    print(result)

    print("\n=== Retry case ===")
    result = await analyze_with_llm_async("RETRY: What is Python?")
    print(result)

    print("\n=== Timeout case ===")
    try:
        result = await analyze_with_llm_async("TIMEOUT: What is Python?")
        print(result)
    except LLMTimeoutError as e:
        print(f"Caught timeout: {e}")
    except LLMRetryError as e:
        print(f"Caught retry error: {e}")


if __name__ == "__main__":
    asyncio.run(main())