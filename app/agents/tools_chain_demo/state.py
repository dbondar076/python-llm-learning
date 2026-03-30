from typing import TypedDict

from app.services.rag_retrieval_service import ScoredChunk


class ChainState(TypedDict, total=False):
    question: str
    messages: list
    top_chunks: list[ScoredChunk]
    answer: str
    route: str