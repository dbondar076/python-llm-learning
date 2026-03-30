def after_retrieve(state: dict) -> str:
    chunks = state.get("top_chunks", [])
    if chunks:
        return "answer"
    return "fallback"