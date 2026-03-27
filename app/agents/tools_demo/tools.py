import ast
import operator as op

from app.services.llm_service import run_text_prompt_with_retry_async
from app.services.rag_retrieval_service import retrieve_top_chunks
from app.services.rag_index_service import ChunkEmbeddingRecord


_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


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


def search_chunks_tool(
    question: str,
    records: list[ChunkEmbeddingRecord],
    top_k: int = 3,
) -> list[dict]:
    chunks = retrieve_top_chunks(
        query=question,
        records=records,
        top_k=top_k,
    )

    return chunks


async def route_tool_with_llm(question: str) -> str:
    prompt = (
        "You are a tool routing assistant.\n"
        "Decide how the system should handle the user's message.\n\n"
        "Return exactly one word:\n"
        "- calculator -> if the user wants to calculate, evaluate, or compute a math expression\n"
        "- list_docs -> if the user asks what documents, sources, or knowledge base files are available\n"
        "- search_chunks -> if the user asks a factual question that should be answered by searching the knowledge base\n"
        "- direct -> for greetings, general chat, or questions that do not require a tool\n\n"
        "Rules:\n"
        "- Use calculator for arithmetic questions and numeric expressions\n"
        "- Use list_docs when the user asks about available documents or sources\n"
        "- Use search_chunks for factual knowledge-base lookup\n"
        "- Use direct for greetings, small talk, and non-tool questions\n"
        "- Do not explain your choice\n\n"
        f"User message:\n{question}"
    )

    raw_result = await run_text_prompt_with_retry_async(prompt)
    route = raw_result.strip().lower()

    if route not in {"calculator", "list_docs", "search_chunks", "direct"}:
        return "direct"

    return route