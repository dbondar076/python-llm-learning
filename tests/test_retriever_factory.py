import pytest

from app.services.rag_index_service import ChunkEmbeddingRecord
from app.services.retrievers.chroma_retriever import ChromaRetriever
from app.services.retrievers.local_retriever import LocalRetriever
from app.services.retrievers import factory


def _dummy_records():
    return [
        ChunkEmbeddingRecord(
            chunk_id="1",
            doc_id="doc1",
            title="Python basics",
            text="Python is a programming language",
            embedding=[0.1, 0.2, 0.3],
        )
    ]


def test_factory_returns_local_retriever(monkeypatch):
    monkeypatch.setattr(factory, "VECTOR_BACKEND", "local")

    retriever = factory.build_retriever(_dummy_records())

    assert isinstance(retriever, LocalRetriever)


def test_factory_returns_chroma_retriever(monkeypatch, tmp_path):
    monkeypatch.setattr(factory, "VECTOR_BACKEND", "chroma")
    monkeypatch.setattr(factory, "CHROMA_COLLECTION_NAME", "test_chunks")
    monkeypatch.setattr(factory, "CHROMA_PERSIST_DIR", str(tmp_path))

    retriever = factory.build_retriever(_dummy_records())

    assert isinstance(retriever, ChromaRetriever)


def test_factory_invalid_backend(monkeypatch):
    monkeypatch.setattr(factory, "VECTOR_BACKEND", "unknown")

    with pytest.raises(ValueError, match="Unsupported VECTOR_BACKEND"):
        factory.build_retriever(_dummy_records())