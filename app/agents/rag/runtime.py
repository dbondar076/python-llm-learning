import uuid

from langchain_core.messages import HumanMessage

from app.agents.rag.graph import get_agent_graph
from app.agents.rag.state import GraphState
from app.services.agent_runtime import build_memory_from_messages, resolve_question_with_memory
from app.services.rag_index_service import ChunkEmbeddingRecord
from app.settings import RAG_TOP_K, RAG_MIN_SCORE


async def run_langgraph_agent(
    question: str,
    records: list[ChunkEmbeddingRecord],
    session_id: str | None = None,
    top_k: int = RAG_TOP_K,
    min_score: float = RAG_MIN_SCORE,
    title_filter: str | None = None,
    doc_id_filter: str | None = None,
) -> GraphState:
    graph = get_agent_graph()

    original_question = question
    memory = None
    last_route = None

    if session_id:
        thread_id = session_id
    else:
        thread_id = f"tmp-{uuid.uuid4()}"

    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await graph.aget_state(config)

    if snapshot and snapshot.values:
        values = snapshot.values
        messages = values.get("messages", [])
        memory = build_memory_from_messages(messages)
        last_route = values.get("route")

    question = await resolve_question_with_memory(
        question=question,
        memory=memory,
        last_route=last_route,
    )

    result = await graph.ainvoke(
        {
            "question": question,
            "original_question": original_question,
            "messages": [HumanMessage(content=original_question)],
            "records": records,
            "top_k": top_k,
            "min_score": min_score,
            "title_filter": title_filter,
            "doc_id_filter": doc_id_filter,
        },
        config=config,
    )

    return result


async def prepare_langgraph_stream(
    question: str,
    records: list[ChunkEmbeddingRecord],
    session_id: str | None = None,
    top_k: int = RAG_TOP_K,
    min_score: float = RAG_MIN_SCORE,
    title_filter: str | None = None,
    doc_id_filter: str | None = None,
):
    graph = get_agent_graph()

    original_question = question
    memory = None
    last_route = None

    if session_id:
        thread_id = session_id
    else:
        thread_id = f"tmp-{uuid.uuid4()}"

    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await graph.aget_state(config)

    if snapshot and snapshot.values:
        values = snapshot.values
        messages = values.get("messages", [])
        memory = build_memory_from_messages(messages)
        last_route = values.get("route")

    question = await resolve_question_with_memory(
        question=question,
        memory=memory,
        last_route=last_route,
    )

    initial_state = {
        "question": question,
        "original_question": original_question,
        "messages": [HumanMessage(content=original_question)],
        "records": records,
        "top_k": top_k,
        "min_score": min_score,
        "title_filter": title_filter,
        "doc_id_filter": doc_id_filter,
    }

    return graph, config, initial_state


