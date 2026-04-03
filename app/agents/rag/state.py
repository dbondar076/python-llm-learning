from langgraph.graph import MessagesState
from app.services.rag_index_service import ChunkEmbeddingRecord
from app.services.rag_retrieval_service import ScoredChunk


class GraphState(MessagesState):
    question: str
    original_question: str
    messages: list
    initial_route: str
    route: str
    top_chunks: list[ScoredChunk]
    answer: str
    records: list[ChunkEmbeddingRecord]
    top_k: int
    min_score: float
    title_filter: str | None
    doc_id_filter: str | None
    retrieval_confidence: float
    retrieval_can_answer: bool