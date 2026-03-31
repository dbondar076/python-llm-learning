from typing import Protocol

from app.services.rag_retrieval_service import ScoredChunk


class Retriever(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 3,
        title_filter: str | None = None,
        doc_id_filter: str | None = None,
    ) -> list[ScoredChunk]:
        ...