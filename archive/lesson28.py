import asyncio
import time


semaphore = asyncio.Semaphore(3)


async def mock_task(task_id: int) -> str:
    async with semaphore:
        print(f"Start {task_id}")
        await asyncio.sleep(1)
        print(f"End {task_id}")
        return f"Task {task_id} done"


async def main() -> None:
    start = time.perf_counter()

    tasks = [mock_task(i) for i in range(1, 11)]
    results = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start

    print(results)
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())