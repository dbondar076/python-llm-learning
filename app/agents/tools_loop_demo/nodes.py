import logging

from app.agents.tools_loop_demo.registry import is_known_tool
from app.agents.tools_loop_demo.state import ToolsLoopState
from app.agents.tools_loop_demo.tools import (
    calculator_tool,
    decide_next_tool_with_llm,
    extract_expression,
    list_documents_tool,
    search_chunks_tool,
    assess_whether_to_continue_with_llm,
)
from app.services.llm_service import run_text_prompt_with_retry_async

logger = logging.getLogger(__name__)


async def decide_node(state: ToolsLoopState) -> ToolsLoopState:
    steps_taken = state.get("steps_taken", 0)
    max_steps = state.get("max_steps", 2)

    if steps_taken >= max_steps:
        return {
            "selected_tool": "finish",
        }

    selected_tool = await decide_next_tool_with_llm(
        question=state["question"],
        steps_taken=steps_taken,
        max_steps=max_steps,
        previous_tool_output=state.get("tool_output"),
    )

    tool_input = state["question"]

    if selected_tool != "finish" and not is_known_tool(selected_tool):
        selected_tool = "finish"

    if selected_tool == "calculator":
        expression = extract_expression(state["question"])
        if expression:
            tool_input = expression
        else:
            selected_tool = "finish"

    return {
        "selected_tool": selected_tool,
        "tool_input": tool_input,
    }


async def calculator_node(state: ToolsLoopState) -> ToolsLoopState:
    output = calculator_tool(state["tool_input"])

    history = list(state.get("history", []))
    history.append(
        {
            "tool": "calculator",
            "input": state["tool_input"],
            "output": output,
        }
    )

    logger.info(
        "LOOP step=%s tool=%s input=%s",
        state.get("steps_taken", 0) + 1,
        "calculator",
        state["tool_input"],
    )

    return {
        "tool_output": output,
        "steps_taken": state.get("steps_taken", 0) + 1,
        "history": history,
    }


async def search_node(state: ToolsLoopState) -> ToolsLoopState:
    output = search_chunks_tool(
        question=state["tool_input"],
        records=state.get("records", []),
        top_k=state.get("top_k", 3),
    )

    history = list(state.get("history", []))
    history.append(
        {
            "tool": "search_chunks",
            "input": state["tool_input"],
            "output": output,
        }
    )

    logger.info(
        "LOOP step=%s tool=%s input=%s",
        state.get("steps_taken", 0) + 1,
        "search_chunks",
        state["tool_input"],
    )

    return {
        "tool_output": output,
        "steps_taken": state.get("steps_taken", 0) + 1,
        "history": history,
    }


def build_history_text(history: list[dict]) -> str:
    if not history:
        return ""

    parts = []
    for i, step in enumerate(history, start=1):
        parts.append(
            f"Step {i}\n"
            f"Tool: {step.get('tool', '')}\n"
            f"Input: {step.get('input', '')}\n"
            f"Output:\n{step.get('output', '')}"
        )

    return "\n\n".join(parts)


async def finish_node(state: ToolsLoopState) -> ToolsLoopState:
    question = state["question"]
    history = state.get("history", [])
    history_text = build_history_text(history)

    if not history_text:
        return {
            "answer": "I don't know based on the available information.",
        }

    prompt = (
        "You are preparing the final answer for the user.\n"
        "Use the full tool history below.\n"
        "Be concise, helpful, and grounded in the tool outputs.\n"
        "Do not invent facts.\n\n"
        f"Question:\n{question}\n\n"
        f"Tool history:\n{history_text}\n\n"
        "Final answer:"
    )

    answer = await run_text_prompt_with_retry_async(prompt)

    return {
        "answer": answer,
    }


async def list_docs_node(state: ToolsLoopState) -> ToolsLoopState:
    output = list_documents_tool(state.get("records", []))

    history = list(state.get("history", []))
    history.append(
        {
            "tool": "list_docs",
            "input": "records",
            "output": output,
        }
    )

    logger.info(
        "LOOP step=%s tool=%s input=%s",
        state.get("steps_taken", 0) + 1,
        "list_docs",
        "records",
    )

    return {
        "tool_output": output,
        "steps_taken": state.get("steps_taken", 0) + 1,
        "history": history,
    }


async def assess_node(state: ToolsLoopState) -> ToolsLoopState:
    steps_taken = state.get("steps_taken", 0)
    max_steps = state.get("max_steps", 2)

    if steps_taken >= max_steps:
        return {
            "next_action": "finish",
        }

    history = state.get("history", [])
    history_text = build_history_text(history)

    if not history_text:
        return {
            "next_action": "finish",
        }

    decision = await assess_whether_to_continue_with_llm(
        question=state["question"],
        history_text=history_text,
        steps_taken=steps_taken,
        max_steps=max_steps,
    )

    return {
        "next_action": decision,
    }