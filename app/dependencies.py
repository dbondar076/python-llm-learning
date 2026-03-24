from fastapi import Request

from app.services.rag_index_service import ChunkEmbeddingRecord


def get_rag_records(request: Request) -> list[ChunkEmbeddingRecord]:
    records = getattr(request.app.state, "rag_records", None)

    if records is None:
        raise RuntimeError("RAG index is not loaded")

    return records