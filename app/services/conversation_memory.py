from typing import TypedDict


class ConversationState(TypedDict, total=False):
    last_user_message: str
    last_agent_route: str
    last_agent_answer: str


_MEMORY_STORE: dict[str, ConversationState] = {}


def get_conversation_state(session_id: str) -> ConversationState | None:
    return _MEMORY_STORE.get(session_id)


def save_conversation_state(session_id: str, state: ConversationState) -> None:
    _MEMORY_STORE[session_id] = state


def clear_conversation_state(session_id: str) -> None:
    _MEMORY_STORE.pop(session_id, None)


def reset_memory_store() -> None:
    _MEMORY_STORE.clear()