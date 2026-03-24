import time
from typing import TypedDict


from app.settings import CACHE_TTL_SECONDS


class CacheEntry(TypedDict):
    value: str
    expires_at: float


TEXT_CACHE: dict[str, CacheEntry] = {}
JSON_CACHE: dict[str, CacheEntry] = {}


def get_cached_value(cache: dict[str, CacheEntry], key: str) -> str | None:
    entry = cache.get(key)

    if entry is None:
        return None

    if time.time() > entry["expires_at"]:
        del cache[key]
        return None

    return entry["value"]


def set_cached_value(cache: dict[str, CacheEntry], key: str, value: str) -> None:
    cache[key] = {
        "value": value,
        "expires_at": time.time() + CACHE_TTL_SECONDS,
    }