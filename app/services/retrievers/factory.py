from app.services.rag_index_service import ChunkEmbeddingRecord
from app.services.retrievers.local_retriever import LocalRetriever


def build_retriever(records: list[ChunkEmbeddingRecord]):
    return LocalRetriever(records)