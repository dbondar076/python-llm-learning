import json
import logging

from app.services.llm_service import run_json_prompt_with_retry_async
from app.services.rag_answer_service import build_context, merge_adjacent_chunks
from app.services.rag_retrieval_service import ScoredChunk

logger = logging.getLogger(__name__)


ANSWERABILITY_SCHEMA = {
    "type": "object",
    "properties": {
        "can_answer": {
            "type": "boolean",
        },
        "reason": {
            "type": "string",
        },
    },
    "required": ["can_answer", "reason"],
    "additionalProperties": False,
}


def build_answerability_prompt(question: str, chunks: list[ScoredChunk]) -> str:
    merged_chunks = merge_adjacent_chunks(chunks)
    context = build_context(merged_chunks)

    return (
        "You are validating whether a user's question can be answered strictly from retrieved context.\n\n"
        "Return JSON with:\n"
        "- can_answer: true or false\n"
        "- reason: short explanation\n\n"
        "Rules:\n"
        "- true ONLY if the answer is explicitly present in the context.\n"
        "- false if the context only mentions the entity but not the requested field.\n"
        "- false if the context is about a similar but different entity.\n"
        "- false if answering would require guessing, outside knowledge, or inference.\n"
        "- false for missing attributes like rating, budget, author, leader, date, etc. when not explicitly stated.\n"
        "- false for unknown or invented entities not explicitly resolved by the context.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}"
    )


async def judge_retrieval_answerability(
    question: str,
    chunks: list[ScoredChunk],
) -> bool:
    if not chunks:
        return False

    prompt = build_answerability_prompt(question, chunks)
    raw_result = await run_json_prompt_with_retry_async(prompt, ANSWERABILITY_SCHEMA)

    try:
        data = json.loads(raw_result)
    except json.JSONDecodeError:
        logger.warning("Failed to parse answerability JSON: %r", raw_result)
        return False

    can_answer = bool(data.get("can_answer", False))
    reason = data.get("reason", "")

    logger.info(
        "ANSWERABILITY judge: can_answer=%s reason=%r question=%r",
        can_answer,
        reason,
        question,
    )

    return can_answer