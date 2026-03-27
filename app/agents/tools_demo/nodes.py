import re

from app.agents.tools_demo.state import ToolDemoState
from app.agents.tools_demo.tools import (
    calculator_tool,
    list_documents_tool,
    route_tool_with_llm,
    search_chunks_tool,
)
from app.services.rag_tools import direct_answer_tool
from app.services.llm_service import run_text_prompt_with_retry_async


_CALC_EXTRACT_PATTERN = re.compile(r"[\d\.\+\-\*\/\(\)\s]+")


def extract_expression(text: str) -> str | None:
    matches = _CALC_EXTRACT_PATTERN.findall(text)

    for m in matches:
        candidate = m.strip()
        if any(op in candidate for op in "+-*/"):
            return candidate

    return None


async def decide_tool_node(state: ToolDemoState) -> ToolDemoState:
    question = state["question"].strip()

    route = await route_tool_with_llm(question)

    if route == "calculator":
        expression = extract_expression(question)

        if expression:
            return {
                "route": "tool",
                "selected_tool": "calculator",
                "tool_input": expression,
            }

        return {
            "route": "direct",
        }

    if route == "list_docs":
        return {
            "route": "tool",
            "selected_tool": "list_docs",
        }

    if route == "search_chunks":
        return {
            "route": "tool",
            "selected_tool": "search_chunks",
            "tool_input": question,
        }

    return {
        "route": "direct",
    }


async def calculator_node(state: ToolDemoState) -> ToolDemoState:
    expression = state["tool_input"]
    output = calculator_tool(expression)

    return {
        "tool_output": output,
    }


async def respond_with_tool_node(state: ToolDemoState) -> ToolDemoState:
    expression = state.get("tool_input", "")
    output = state.get("tool_output", "")

    return {
        "answer": f"{expression} = {output}",
    }


async def direct_answer_node(state: ToolDemoState) -> ToolDemoState:
    answer = await direct_answer_tool(state["question"])
    return {
        "answer": answer,
    }


async def list_docs_node(state: ToolDemoState) -> ToolDemoState:
    output = list_documents_tool(state.get("records", []))

    return {
        "tool_output": output,
    }


async def respond_with_docs_node(state: ToolDemoState) -> ToolDemoState:
    output = state.get("tool_output", "")

    return {
        "answer": f"Available documents: {output}",
    }


async def search_chunks_node(state: ToolDemoState) -> ToolDemoState:
    chunks = search_chunks_tool(
        question=state.get("tool_input", state["question"]),
        records=state.get("records", []),
        top_k=state.get("top_k", 3),
    )

    return {
        "tool_output": chunks,
    }


async def respond_with_search_node(state: ToolDemoState) -> ToolDemoState:
    question = state["question"]
    chunks = state.get("tool_output", [])

    if not chunks:
        return {
            "answer": "I don't know based on the provided context."
        }

    context = "\n\n".join(
        f"{c['title']}: {c['text']}"
        for c in chunks
    )

    prompt = (
        "Answer the question using ONLY the provided context.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )

    answer = await run_text_prompt_with_retry_async(prompt)

    return {
        "answer": answer,
    }