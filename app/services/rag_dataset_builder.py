from app.services.rag_index_service import ChunkEmbeddingRecord
from app.services.rag_retrieval_service import get_query_embedding


def build_records_from_documents(documents: list[dict]) -> list[ChunkEmbeddingRecord]:
    records: list[ChunkEmbeddingRecord] = []

    for doc in documents:
        records.append(
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "chunk_id": f'{doc["doc_id"]}_chunk_1',
                "text": doc["text"],
                "embedding": get_query_embedding(doc["text"]),
            }
        )

    return records