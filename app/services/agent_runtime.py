import logging

from langchain_core.messages import AIMessage, HumanMessage
from app.services.rag_tools import rewrite_question_with_memory_tool

logger = logging.getLogger(__name__)


# ----------------------------
# Rewrite guardrails
# ----------------------------

def is_too_generic_rewrite(text: str) -> bool:
    text = text.lower().strip()

    return (
        text.startswith("information about")
        or text.startswith("details about")
        or len(text.split()) < 2
    )


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

    lowered = text.lower()

    technical_aliases = {
        "python": "Python programming language",
        "fastapi": "FastAPI Python framework",
        "api": "API development",
        "llm": "large language models",
        "ai": "artificial intelligence",
    }

    if lowered in technical_aliases:
        return technical_aliases[lowered]

    if len(text.split()) == 1:
        return f"{text} technical topic"

    return text


# ----------------------------
# Rewrite with memory
# ----------------------------

async def resolve_question_with_memory(
    question: str,
    memory: dict | None,
    last_route: str | None = None,
) -> str:
    if not memory:
        logger.info("RESOLVE: no memory")
        return question

    last_user_message = memory.get("last_user_message")
    last_agent_answer = memory.get("last_agent_answer")

    logger.info(
        "RESOLVE memory: user=%r ai=%r route=%r",
        last_user_message,
        last_agent_answer,
        last_route,
    )

    if last_route == "clarify" and last_user_message:
        rewritten = await rewrite_question_with_memory_tool(
            previous_user_message=last_user_message,
            previous_agent_answer=last_agent_answer,
            current_user_message=question,
        )

        logger.info("Rewrite candidate: %r", rewritten)

        if (
            not rewritten
            or is_bad_rewrite(rewritten)
            or is_too_generic_rewrite(rewritten)
        ):
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
# Memorization
# ----------------------------

def build_memory_from_messages(messages: list) -> dict | None:
    if not messages:
        return None

    last_human = None
    last_ai = None

    for message in reversed(messages):
        if last_human is None and isinstance(message, HumanMessage):
            last_human = message
        elif last_ai is None and isinstance(message, AIMessage):
            last_ai = message

        if last_human is not None and last_ai is not None:
            break

    if last_human is None and last_ai is None:
        return None

    return {
        "last_user_message": last_human.content if last_human else None,
        "last_agent_answer": last_ai.content if last_ai else None,
    }