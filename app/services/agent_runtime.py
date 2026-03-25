import logging

from app.services.conversation_memory import save_conversation_state
from app.services.rag_tools import rewrite_question_with_memory_tool

logger = logging.getLogger(__name__)


# ----------------------------
# Rewrite guardrails
# ----------------------------

def is_bad_rewrite(text: str) -> bool:
    lowered = text.lower().strip()

    bad_patterns = [
        "clarify",
        "refers to",
        "refer to",
        "reference to",
        "previous message",
        "user message",
        "current user message",
        "when the user",
        "the user replies",
        "the user replied",
        "meaning of",
        "in the context of",
        "what does",
        "what do you mean",
    ]

    return any(p in lowered for p in bad_patterns)


def build_fallback_rewrite(current_user_message: str) -> str:
    text = current_user_message.strip()
    if not text:
        return current_user_message
    return f"Information about {text}"


# ----------------------------
# Rewrite with memory
# ----------------------------

async def resolve_question_with_memory(
    question: str,
    memory: dict | None,
) -> str:
    if not memory:
        logger.info("RESOLVE: no memory")
        return question

    last_user_message = memory.get("last_user_message")
    last_agent_answer = memory.get("last_agent_answer")
    last_route = memory.get("last_agent_route")

    logger.info("RESOLVE memory: %s", memory)

    if last_route == "clarify" and last_user_message:
        rewritten = await rewrite_question_with_memory_tool(
            previous_user_message=last_user_message,
            previous_agent_answer=last_agent_answer,
            current_user_message=question,
        )

        logger.info("Rewrite candidate: %r", rewritten)

        if not rewritten or is_bad_rewrite(rewritten):
            fallback = build_fallback_rewrite(question)
            logger.info("Rejected rewrite → fallback=%r", fallback)
            return fallback

        logger.info("Accepted rewrite=%r", rewritten)
        return rewritten

    return question


# ----------------------------
# Routing helper
# ----------------------------

def should_force_rag_for_resolved_question(question: str) -> bool:
    normalized = question.lower()

    keywords = [
        "python",
        "api",
        "framework",
        "language",
        "fastapi",
        "ai",
        "data",
    ]

    return any(k in normalized for k in keywords)


# ----------------------------
# Memory saving
# ----------------------------

def save_memory_if_needed(
    session_id: str | None,
    original_question: str,
    state: dict,
) -> None:
    if not session_id:
        return

    save_conversation_state(
        session_id,
        {
            "last_user_message": original_question,
            "last_agent_route": state.get("route"),
            "last_agent_answer": state.get("answer"),
        },
    )