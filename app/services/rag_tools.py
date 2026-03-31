import logging
from langchain_core.messages import BaseMessage, HumanMessage
from app.settings import RAG_TOP_K
from app.services.llm_service import run_text_prompt_with_retry_async
from app.services.rag_answer_service import (
    build_context,
    build_rag_prompt,
    merge_adjacent_chunks,
)
from app.services.rag_index_service import ChunkEmbeddingRecord
from app.services.rag_retrieval_service import ScoredChunk, retrieve_top_chunks


logger = logging.getLogger(__name__)


# ----------------------------
# RETRIEVAL TOOL
# ----------------------------

async def search_chunks_tool(
    question: str,
    retriever,
    top_k: int = RAG_TOP_K,
    title_filter: str | None = None,
    doc_id_filter: str | None = None,
) -> list[ScoredChunk]:
    return retriever.search(
        query=question,
        top_k=top_k,
        title_filter=title_filter,
        doc_id_filter=doc_id_filter,
    )


# ----------------------------
# RAG ANSWER TOOL
# ----------------------------

async def generate_grounded_answer_tool(
    messages: list[BaseMessage],
    chunks: list[ScoredChunk],
) -> str:
    merged_chunks = merge_adjacent_chunks(chunks)
    context = build_context(merged_chunks)

    question = get_last_human_message_text(messages)
    prompt = build_rag_prompt(question, context)

    return await run_text_prompt_with_retry_async(prompt)


# ----------------------------
# DIRECT ANSWER TOOL
# ----------------------------

async def direct_answer_tool(question: str) -> str:
    prompt = (
        "Answer the user's message directly.\n"
        "Be brief, natural, and helpful.\n"
        "If it is a greeting, respond like a polite assistant.\n"
        "Do not mention missing context.\n\n"
        f"User message:\n{question}"
    )
    return await run_text_prompt_with_retry_async(prompt)


# ----------------------------
# CLARIFY TOOL
# ----------------------------

async def clarify_question_tool(question: str) -> str:
    prompt = (
        "The user's message is too vague or ambiguous.\n"
        "Ask one short clarifying question.\n"
        "Be polite and concise.\n"
        "Do not answer the original question yet.\n\n"
        f"User message:\n{question}"
    )
    return await run_text_prompt_with_retry_async(prompt)


# ----------------------------
# ROUTER
# ----------------------------

async def route_question_with_llm(question: str) -> str:
    prompt = (
        "You are a routing assistant.\n"
        "Decide how the system should handle the user's message.\n\n"
        "Return exactly one word:\n"
        "- direct -> greetings, small talk, simple conversational input\n"
        "- rag -> factual questions requiring knowledge\n"
        "- clarify -> vague, incomplete, or ambiguous questions\n\n"
        "Rules:\n"
        "- If unclear → clarify\n"
        "- If small talk → direct\n"
        "- If knowledge needed → rag\n"
        "Do not explain your choice.\n\n"
        f"User message:\n{question}"
    )

    raw_result = await run_text_prompt_with_retry_async(prompt)
    route = raw_result.strip().lower()

    if route not in {"direct", "rag", "clarify"}:
        return "direct"

    return route


async def rewrite_question_with_memory_tool(
    previous_user_message: str,
    previous_agent_answer: str | None,
    current_user_message: str,
) -> str:
    prompt = (
        "Rewrite the current user message into a standalone factual query using the previous conversation context.\n"
        "The rewritten query will be used for semantic search in a technical knowledge base about software, programming, APIs, and AI.\n\n"
        "Rules:\n"
        "- Use only information present in the conversation\n"
        "- Do NOT invent missing context\n"
        "- Do NOT ask the user a question\n"
        "- Do NOT be conversational\n"
        "- Keep the rewrite short and literal\n"
        "- Return ONLY the rewritten query\n\n"
        f"Previous user message:\n{previous_user_message}\n\n"
        f"Previous agent message:\n{previous_agent_answer or ''}\n\n"
        f"Current user message:\n{current_user_message}"
    )

    result = await run_text_prompt_with_retry_async(prompt)

    logger.info(
        "Rewrite tool result: prev_user=%r prev_agent=%r current=%r rewritten=%r",
        previous_user_message,
        previous_agent_answer,
        current_user_message,
        result,
    )

    return result


# ----------------------------
# HELPERS
# ----------------------------

def get_last_human_message_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content

    return ""