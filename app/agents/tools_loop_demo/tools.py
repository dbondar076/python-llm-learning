import ast
import operator as op
import re

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
        lines.append(
            f"{chunk['title']} [{chunk['chunk_id']}]: {chunk['text']}"
        )

    return "\n".join(lines)


async def decide_next_tool_with_llm(
    question: str,
    steps_taken: int,
    max_steps: int,
    previous_tool_output: str | None,
) -> str:
    prompt = (
        "You are a tool-loop routing assistant.\n"
        "Decide the next action.\n\n"
        "Return exactly one word:\n"
        "- calculator\n"
        "- search_chunks\n"
        "- list_docs\n"
        "- finish\n\n"
        "Rules:\n"
        "- Use calculator for math questions.\n"
        "- Use search_chunks for factual knowledge-base questions.\n"
        "- Use list_docs when the user asks what documents or sources are available.\n"
        "- Use finish if enough information is already available or max steps are reached.\n"
        "- Do not explain your choice.\n\n"
        f"Question:\n{question}\n\n"
        f"Steps taken: {steps_taken}\n"
        f"Max steps: {max_steps}\n\n"
        f"Previous tool output:\n{previous_tool_output or ''}"
    )

    raw = await run_text_prompt_with_retry_async(prompt)
    route = raw.strip().lower()

    allowed = {"calculator", "search_chunks", "list_docs", "finish"}

    if route not in allowed:
        return "finish"

    return route


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