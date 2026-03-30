import logging

from app.agents.tools_loop_demo.registry import get_tool_config, is_known_tool
from app.agents.tools_loop_demo.state import ToolsLoopState
from app.agents.tools_loop_demo.tool_validation import validate_tool_arguments
from app.agents.tools_loop_demo.tools import (
    assess_whether_to_continue_with_llm,
    decide_next_tool_with_llm,
    extract_expression,
)
from app.services.llm_service import run_text_prompt_with_retry_async

logger = logging.getLogger(__name__)


async def decide_node(state: ToolsLoopState) -> ToolsLoopState:
    steps_taken = state.get("steps_taken", 0)
    max_steps = state.get("max_steps", 2)

    if steps_taken >= max_steps:
        return {
            "selected_tool": "finish",
            "decision_reason": "Max steps reached",
            "tool_input": state["question"],
            "tool_arguments": {},
        }

    decision = await decide_next_tool_with_llm(
        question=state["question"],
        steps_taken=steps_taken,
        max_steps=max_steps,
        previous_tool_output=state.get("tool_output"),
    )

    selected_tool = decision.tool
    decision_reason = decision.reason
    tool_arguments = dict(decision.arguments or {})

    if selected_tool != "finish" and not is_known_tool(selected_tool):
        return {
            "selected_tool": "finish",
            "decision_reason": "Unknown tool",
            "tool_input": state["question"],
            "tool_arguments": {},
        }

    is_valid, validated_args = validate_tool_arguments(
        selected_tool,
        tool_arguments,
    )

    if not is_valid:
        selected_tool = "finish"
        decision_reason = "Invalid tool arguments"
        tool_arguments = {}
    else:
        tool_arguments = validated_args

    tool_input = state["question"]

    if selected_tool == "calculator":
        expression = tool_arguments.get("expression")

        if not expression:
            extracted = extract_expression(state["question"])
            if extracted:
                tool_arguments["expression"] = extracted
                tool_input = extracted
            else:
                selected_tool = "finish"
                decision_reason = "Could not determine calculator expression"
                tool_arguments = {}
        else:
            tool_input = expression

    return {
        "selected_tool": selected_tool,
        "decision_reason": decision_reason,
        "tool_input": tool_input,
        "tool_arguments": tool_arguments,
    }


def build_history_text(history: list[dict]) -> str:
    if not history:
        return ""

    parts = []
    for i, step in enumerate(history, start=1):
        parts.append(
            f"Step {i}\n"
            f"Tool: {step.get('tool', '')}\n"
            f"Reason: {step.get('reason', '')}\n"
            f"Arguments: {step.get('arguments', {})}\n"
            f"Input: {step.get('input', '')}\n"
            f"Output:\n{step.get('output', '')}"
        )

    return "\n\n".join(parts)


async def tool_node(state: ToolsLoopState) -> ToolsLoopState:
    selected_tool = state.get("selected_tool")
    config = get_tool_config(selected_tool or "")

    if not config:
        return {
            "tool_output": "",
        }

    input_mode = config["input_mode"]
    tool_callable = config["callable"]
    tool_arguments = dict(state.get("tool_arguments", {}))

    if input_mode == "tool_input":
        expression = tool_arguments.get("expression", "")
        output = tool_callable(expression)
        history_input = expression

    elif input_mode == "records":
        output = tool_callable(state.get("records", []))
        history_input = "records"

    elif input_mode == "question+records":
        question = state["question"]
        output = tool_callable(
            question=question,
            records=state.get("records", []),
            top_k=state.get("top_k", 3),
        )
        history_input = question

    else:
        output = ""
        history_input = ""

    history = list(state.get("history", []))
    history.append(
        {
            "tool": selected_tool,
            "reason": state.get("decision_reason", ""),
            "arguments": tool_arguments,
            "input": history_input,
            "output": output,
        }
    )

    logger.info(
        "LOOP step=%s tool=%s input=%s",
        state.get("steps_taken", 0) + 1,
        selected_tool,
        history_input,
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