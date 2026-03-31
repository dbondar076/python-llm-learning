from app.services.rag_index_service import ChunkEmbeddingRecord
from app.services.retrievers.chroma_retriever import ChromaRetriever
from app.services.retrievers.local_retriever import LocalRetriever
from app.settings import (
    VECTOR_BACKEND,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
)


def build_retriever(records: list[ChunkEmbeddingRecord]):
    if VECTOR_BACKEND == "local":
        return LocalRetriever(records)

    if VECTOR_BACKEND == "chroma":
        return ChromaRetriever(
            records=records,
            collection_name=CHROMA_COLLECTION_NAME,
            persist_dir=CHROMA_PERSIST_DIR,
        )

    raise ValueError(f"Unsupported VECTOR_BACKEND: {VECTOR_BACKEND}")