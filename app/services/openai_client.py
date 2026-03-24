import app.settings as settings

from openai import OpenAI


_OPENAI_CLIENT: OpenAI | None = None


def get_openai_client() -> OpenAI:
    global _OPENAI_CLIENT

    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")

    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = OpenAI(api_key=settings.OPENAI_API_KEY)

    return _OPENAI_CLIENT


def reset_openai_client() -> None:
    global _OPENAI_CLIENT
    _OPENAI_CLIENT = None