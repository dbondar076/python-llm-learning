import json
from typing import TypedDict
from app.settings import CHUNK_EMBEDDINGS_PATH


class ChunkEmbeddingRecord(TypedDict):
    doc_id: str
    title: str
    chunk_id: str
    text: str
    embedding: list[float]


def load_chunk_embeddings(file_path=None):
    path = file_path or str(CHUNK_EMBEDDINGS_PATH)

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)