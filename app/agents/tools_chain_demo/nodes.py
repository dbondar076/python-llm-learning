from langchain_core.messages import AIMessage

from app.services.rag_tools import (
    search_chunks_tool,
    generate_grounded_answer_tool,
)


async def retrieve_node(state: dict, records, top_k: int):
    chunks = await search_chunks_tool(
        question=state["question"],
        records=records,
        top_k=top_k,
    )

    return {
        "top_chunks": chunks,
        "route": "retrieved",
    }


async def answer_node(state: dict):
    answer = await generate_grounded_answer_tool(
        messages=state.get("messages", []),
        chunks=state.get("top_chunks", []),
    )

    messages = list(state.get("messages", []))
    messages.append(AIMessage(content=answer))

    return {
        "answer": answer,
        "route": "answered",
        "messages": messages,
    }


async def fallback_node(state: dict):
    return {
        "answer": "I don't know based on the provided context.",
        "route": "fallback",
    }