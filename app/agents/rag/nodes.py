from langchain_core.messages import AIMessage

from app.agents.rag.state import GraphState
from app.services.agent_runtime import should_force_rag_for_resolved_question
from app.services.rag_answer_service import NO_ANSWER
from app.services.rag_tools import (
    route_question_with_llm,
    direct_answer_tool,
    clarify_question_tool,
    search_chunks_tool,
    generate_grounded_answer_tool,
)
from app.settings import RAG_TOP_K


async def route_node(state: GraphState) -> GraphState:
    question = state["question"]
    route = await route_question_with_llm(question)
    initial_route = route

    if route == "clarify" and should_force_rag_for_resolved_question(question):
        route = "rag"

    return {
        "route": route,
        "initial_route": initial_route,
        "question": state["question"],
        "original_question": state.get("original_question"),
        "messages": state.get("messages", []),
    }


async def direct_node(state: GraphState) -> GraphState:
    answer = await direct_answer_tool(state["question"])
    messages = list(state.get("messages", []))
    messages.append(AIMessage(content=answer))

    return {
        "answer": answer,
        "top_chunks": [],
        "route": "direct",
        "messages": messages,
    }


async def clarify_node(state: GraphState) -> GraphState:
    answer = await clarify_question_tool(state["question"])
    messages = list(state.get("messages", []))
    messages.append(AIMessage(content=answer))

    return {
        "answer": answer,
        "top_chunks": [],
        "route": "clarify",
        "messages": messages,
    }


async def retrieve_node(state: GraphState) -> GraphState:
    chunks = await search_chunks_tool(
        question=state["question"],
        records=state["records"],
        top_k=state.get("top_k", RAG_TOP_K),
        title_filter=state.get("title_filter"),
        doc_id_filter=state.get("doc_id_filter"),
    )
    return {"top_chunks": chunks}


async def answer_node(state: GraphState) -> GraphState:
    answer = await generate_grounded_answer_tool(
        messages=state.get("messages", []),
        chunks=state.get("top_chunks", []),
    )
    messages = list(state.get("messages", []))
    messages.append(AIMessage(content=answer))

    return {
        "answer": answer,
        "route": "answer",
        "messages": messages,
    }


async def fallback_node(state: GraphState) -> GraphState:
    messages = list(state.get("messages", []))
    messages.append(AIMessage(content=NO_ANSWER))

    return {
        "answer": NO_ANSWER,
        "route": "fallback",
        "messages": messages,
    }
