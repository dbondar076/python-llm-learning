from app.agents.tools_loop_demo.state import ToolsLoopState
from app.agents.tools_loop_demo.tools import (
    calculator_tool,
    decide_next_tool_with_llm,
    extract_expression,
    search_chunks_tool,
)
from app.services.llm_service import run_text_prompt_with_retry_async


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

    return {
        "tool_output": output,
        "steps_taken": state.get("steps_taken", 0) + 1,
    }


async def search_node(state: ToolsLoopState) -> ToolsLoopState:
    output = search_chunks_tool(
        question=state["tool_input"],
        records=state.get("records", []),
        top_k=state.get("top_k", 3),
    )

    return {
        "tool_output": output,
        "steps_taken": state.get("steps_taken", 0) + 1,
    }


async def finish_node(state: ToolsLoopState) -> ToolsLoopState:
    question = state["question"]
    tool_output = state.get("tool_output", "")

    if not tool_output:
        return {
            "answer": "I don't know based on the available information.",
        }

    prompt = (
        "Write a final answer for the user based on the tool output.\n"
        "Be concise and helpful.\n\n"
        f"Question:\n{question}\n\n"
        f"Tool output:\n{tool_output}\n\n"
        "Final answer:"
    )

    answer = await run_text_prompt_with_retry_async(prompt)

    return {
        "answer": answer,
    }