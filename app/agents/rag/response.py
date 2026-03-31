from app.agents.rag.state import GraphState


def build_langgraph_meta(state: GraphState) -> dict:
    top_chunks = state.get("top_chunks", [])
    top_score = top_chunks[0]["score"] if top_chunks else None

    return {
        "initial_route": state.get("initial_route"),
        "final_route": state.get("route"),
        "original_question": state.get("original_question"),
        "resolved_question": state.get("question"),
        "top_score": top_score,
        "chunk_count": len(top_chunks),
        "retrieval_confidence": state.get("retrieval_confidence", 0.0),
    }


def build_langgraph_response(state: GraphState) -> dict:
    raw_chunks = state.get("top_chunks", [])

    public_chunks = [
        {
            "doc_id": c["doc_id"],
            "title": c["title"],
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "score": c["score"],
        }
        for c in raw_chunks
    ]

    return {
        "answer": state.get("answer", ""),
        "chunks": public_chunks,
        "meta": build_langgraph_meta(state),
    }