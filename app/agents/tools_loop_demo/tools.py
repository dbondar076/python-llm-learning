import ast
import json
import operator as op
import re

from app.agents.tools_loop_demo.schemas import AgentAction
from app.agents.tools_loop_demo.tool_definitions import TOOL_DEFINITIONS
from app.agents.tools_loop_demo.tool_specs import TOOLS_SPECS
from app.services.llm_service import run_text_prompt_with_retry_async
from app.services.rag_retrieval_service import retrieve_top_chunks


_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}

_CALC_EXTRACT_PATTERN = re.compile(r"[\d\.\+\-\*\/\(\)\s]+")


def extract_expression(text: str) -> str | None:
    matches = _CALC_EXTRACT_PATTERN.findall(text)

    for match in matches:
        candidate = match.strip()
        if any(op_char in candidate for op_char in "+-*/"):
            return candidate

    return None


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        operator_type = type(node.op)

        if operator_type not in _ALLOWED_OPERATORS:
            raise ValueError("Unsupported operator")

        return _ALLOWED_OPERATORS[operator_type](left, right)

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        operator_type = type(node.op)

        if operator_type not in _ALLOWED_OPERATORS:
            raise ValueError("Unsupported unary operator")

        return _ALLOWED_OPERATORS[operator_type](operand)

    raise ValueError("Unsupported expression")


def calculator_tool(expression: str) -> str:
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _eval_node(parsed.body)

        if result.is_integer():
            return str(int(result))

        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception:
        return "Error: invalid expression"


def search_chunks_tool(question: str, records: list[dict], top_k: int = 3) -> str:
    chunks = retrieve_top_chunks(
        query=question,
        records=records,
        top_k=top_k,
    )

    if not chunks:
        return "No relevant chunks found."

    lines = []
    for chunk in chunks:
        lines.append(f"{chunk['title']} [{chunk['chunk_id']}]: {chunk['text']}")

    return "\n".join(lines)


def list_documents_tool(records: list[dict]) -> str:
    seen = []
    seen_ids = set()

    for record in records:
        doc_id = record["doc_id"]
        title = record["title"]

        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            seen.append(f"{title} ({doc_id})")

    if not seen:
        return "No documents available."

    return ", ".join(seen)


def build_tools_description_text() -> str:
    lines = []
    for name, config in TOOLS_SPECS.items():
        lines.append(f"- {name} -> {config['description']}")
    return "\n".join(lines)


def build_tools_examples_text() -> str:
    parts = []

    for name, config in TOOLS_SPECS.items():
        example = {
            "type": "tool_call",
            "tool_name": name,
            "arguments": config.get("arguments_example", {}),
            "reason": "short reason",
        }
        parts.append(json.dumps(example, ensure_ascii=False))

    finish_example = {
        "type": "finish",
        "tool_name": "finish",
        "arguments": {},
        "reason": "short reason",
    }
    parts.append(json.dumps(finish_example, ensure_ascii=False))

    return "\n".join(parts)


def build_tool_definitions_text() -> str:
    lines = []

    for tool in TOOL_DEFINITIONS:
        lines.append(
            json.dumps(
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["input_schema"],
                },
                ensure_ascii=False,
            )
        )

    return "\n".join(lines)


async def decide_next_tool_with_llm(
    question: str,
    steps_taken: int,
    max_steps: int,
    previous_tool_output: str | None,
) -> AgentAction:
    tools_description = build_tools_description_text()
    tools_examples = build_tools_examples_text()
    tool_definitions = build_tool_definitions_text()
    tool_names = list(TOOLS_SPECS.keys())

    allowed_tool_names = tool_names + ["finish"]
    allowed_routes_text = "\n".join(f"- {name}" for name in allowed_tool_names)

    prompt = (
        "You are a function-calling style agent router.\n"
        "Decide the next agent action.\n\n"
        "Return valid JSON with this shape:\n"
        '{"type": "<tool_call|finish>", "tool_name": "<allowed tool name>", "arguments": {...}, "reason": "<short reason>"}\n\n'
        "Allowed tool names:\n"
        f"{allowed_routes_text}\n\n"
        "Tool definitions:\n"
        f"{tool_definitions}\n\n"
        "Available tools summary:\n"
        f"{tools_description}\n\n"
        "Examples of valid outputs:\n"
        f"{tools_examples}\n\n"
        "Rules:\n"
        '- If another tool must be used, return {"type": "tool_call", ...}.\n'
        '- If enough information is available or max steps are reached, return {"type": "finish", "tool_name": "finish", "arguments": {}, ...}.\n'
        "- Arguments must match the selected tool input schema.\n"
        "- Return JSON only.\n\n"
        f"Question:\n{question}\n\n"
        f"Steps taken: {steps_taken}\n"
        f"Max steps: {max_steps}\n\n"
        f"Previous tool output:\n{previous_tool_output or ''}"
    )

    raw = await run_text_prompt_with_retry_async(prompt)
    allowed = set(allowed_tool_names)

    try:
        data = json.loads(raw)
        action = AgentAction.model_validate(data)
    except Exception:
        return AgentAction(
            type="finish",
            tool_name="finish",
            arguments={},
            reason="Invalid model output",
        )

    if action.type not in {"tool_call", "finish"}:
        return AgentAction(
            type="finish",
            tool_name="finish",
            arguments={},
            reason="Unknown action type",
        )

    if action.tool_name not in allowed:
        return AgentAction(
            type="finish",
            tool_name="finish",
            arguments={},
            reason="Unknown tool",
        )

    return action


async def assess_whether_to_continue_with_llm(
    question: str,
    history_text: str,
    steps_taken: int,
    max_steps: int,
) -> str:
    prompt = (
        "You are deciding whether the agent should continue using tools or finish.\n\n"
        "Return exactly one word:\n"
        "- continue\n"
        "- finish\n\n"
        "Rules:\n"
        "- Return finish if the tool history already contains enough information to answer the user's question.\n"
        "- Return finish if max steps are reached.\n"
        "- Return continue only if another tool step is clearly needed.\n"
        "- Do not explain your choice.\n\n"
        f"Question:\n{question}\n\n"
        f"Steps taken: {steps_taken}\n"
        f"Max steps: {max_steps}\n\n"
        f"Tool history:\n{history_text}"
    )

    raw = await run_text_prompt_with_retry_async(prompt)
    decision = raw.strip().lower()

    if decision not in {"continue", "finish"}:
        return "finish"

    return decision