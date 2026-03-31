from app.services.rag_index_service import ChunkEmbeddingRecord
from app.services.rag_retrieval_service import ScoredChunk, retrieve_top_chunks


class LocalRetriever:
    def __init__(self, records: list[ChunkEmbeddingRecord]) -> None:
        self.records = records

    def search(
        self,
        query: str,
        top_k: int = 3,
        title_filter: str | None = None,
        doc_id_filter: str | None = None,
    ) -> list[ScoredChunk]:
        return retrieve_top_chunks(
            query=query,
            records=self.records,
            top_k=top_k,
            title_filter=title_filter,
            doc_id_filter=doc_id_filter,
        )