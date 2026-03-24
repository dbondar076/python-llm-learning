import asyncio

from app.services.rag_answer_service import answer_with_rag
from app.services.rag_index_service import load_chunk_embeddings


async def main() -> None:
    records = load_chunk_embeddings()

    questions = [
        "What can Python be used for?",
        "API framework in Python",
        "What is JavaScript used for?",
    ]

    for question in questions:
        chunks, answer = await answer_with_rag(question, records)

        print("QUESTION:", question)
        print("CHUNKS:")
        for chunk in chunks:
            print(f"{round(chunk['score'], 4)} | [{chunk['title']}] {chunk['text']}")
        print("ANSWER:", answer)
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())