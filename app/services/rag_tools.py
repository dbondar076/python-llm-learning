from app.settings import RAG_TOP_K
from app.services.llm_service import run_text_prompt_with_retry_async
from app.services.rag_answer_service import (
    build_context,
    build_rag_prompt,
    merge_adjacent_chunks,
)
from app.services.rag_index_service import ChunkEmbeddingRecord
from app.services.rag_retrieval_service import ScoredChunk, retrieve_top_chunks


# ----------------------------
# RETRIEVAL TOOL
# ----------------------------

async def search_chunks_tool(
    question: str,
    records: list[ChunkEmbeddingRecord],
    top_k: int = RAG_TOP_K,
    title_filter: str | None = None,
    doc_id_filter: str | None = None,
) -> list[ScoredChunk]:
    return retrieve_top_chunks(
        query=question,
        records=records,
        top_k=top_k,
        title_filter=title_filter,
        doc_id_filter=doc_id_filter,
    )


# ----------------------------
# RAG ANSWER TOOL
# ----------------------------

async def generate_grounded_answer_tool(
    question: str,
    chunks: list[ScoredChunk],
) -> str:
    merged_chunks = merge_adjacent_chunks(chunks)
    context = build_context(merged_chunks)
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