from app.services.rag_index_service import ChunkEmbeddingRecord
from app.services.rag_retrieval_service import get_query_embedding


def split_text_into_chunks(text: str) -> list[str]:
    parts = [part.strip() for part in text.split(". ") if part.strip()]
    chunks: list[str] = []

    current: list[str] = []

    for part in parts:
        normalized = part if part.endswith(".") else f"{part}."
        current.append(normalized)

        if len(current) == 2:
            chunks.append(" ".join(current).strip())
            current = []

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def build_chunked_records_from_documents(documents: list[dict]) -> list[ChunkEmbeddingRecord]:
    records: list[ChunkEmbeddingRecord] = []

    for doc in documents:
        chunks = split_text_into_chunks(doc["text"])

        for index, chunk_text in enumerate(chunks, start=1):
            records.append(
                {
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "chunk_id": f'{doc["doc_id"]}_chunk_{index}',
                    "text": chunk_text,
                    "embedding": get_query_embedding(chunk_text),
                }
            )

    return records