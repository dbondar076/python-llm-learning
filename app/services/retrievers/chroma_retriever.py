from chromadb import PersistentClient

from app.services.rag_index_service import ChunkEmbeddingRecord
from app.services.rag_retrieval_service import get_query_embedding


class ChromaRetriever:
    def __init__(
        self,
        records: list[ChunkEmbeddingRecord],
        collection_name: str,
        persist_dir: str,
    ) -> None:
        self.client = PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

        self._load_records(records)

    def _load_records(self, records: list[ChunkEmbeddingRecord]) -> None:
        existing_count = self.collection.count()
        if existing_count > 0:
            return

        ids: list[str] = []
        documents: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict] = []

        for record in records:
            ids.append(record["chunk_id"])
            documents.append(record["text"])
            embeddings.append(record["embedding"])
            metadatas.append(
                {
                    "doc_id": record["doc_id"],
                    "title": record["title"],
                    "chunk_id": record["chunk_id"],
                }
            )

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def _build_where_filter(
        self,
        title_filter: str | None = None,
        doc_id_filter: str | None = None,
    ) -> dict | None:
        conditions = []

        if title_filter:
            conditions.append({"title": title_filter})

        if doc_id_filter:
            conditions.append({"doc_id": doc_id_filter})

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}

    def search(
        self,
        query: str,
        top_k: int = 3,
        title_filter: str | None = None,
        doc_id_filter: str | None = None,
    ) -> list[dict]:
        query_embedding = get_query_embedding(query)
        where = self._build_where_filter(
            title_filter=title_filter,
            doc_id_filter=doc_id_filter,
        )

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        chunks: list[dict] = []

        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            score = 1.0 / (1.0 + distance) if distance is not None else 0.0

            chunks.append(
                {
                    "doc_id": metadata["doc_id"],
                    "title": metadata["title"],
                    "chunk_id": chunk_id,
                    "text": text,
                    "score": score,
                }
            )

        return chunks