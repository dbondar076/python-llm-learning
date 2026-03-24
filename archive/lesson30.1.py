import asyncio
from app.services.llm_service import analyze_with_llm_async


async def main() -> None:
    result = await analyze_with_llm_async("What is Python?")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())