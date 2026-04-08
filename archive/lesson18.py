import requests

from lesson15 import get_api_key


API_URL = "https://httpbin.org/post"


def build_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def build_payload(prompt: str) -> dict:
    return {
        "model": "demo-model",
        "input": prompt,
    }


def ask_llm_http(prompt: str) -> dict | None:
    api_key = get_api_key()

    if api_key is None:
        return None

    headers = build_headers(api_key)
    payload = build_payload(prompt)

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=10,
        )

        if response.status_code != 200:
            print(f"API error: {response.status_code}")
            return None

        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


if __name__ == "__main__":
    result = ask_llm_http("Explain what Python is in one sentence.")
    if result is not None:
        print(result["json"])
        print(result["headers"])


# import requests
# from typing import Any
#
# from lesson15 import get_api_key
#
#
# API_URL = "https://httpbin.org/post"
#
#
# def build_headers(api_key: str) -> dict[str, str]:
#     return {
#         "Authorization": f"Bearer {api_key}",
#         "Content-Type": "application/json",
#     }
#
#
# def build_payload(prompt: str, model: str = "demo-model") -> dict[str, Any]:
#     return {
#         "model": model,
#         "input": prompt,
#     }
#
#
# def ask_llm_http(prompt: str) -> dict[str, Any] | None:
#     api_key = get_api_key()
#     if api_key is None:
#         return None
#
#     try:
#         response = requests.post(
#             API_URL,
#             headers=build_headers(api_key),
#             json=build_payload(prompt),
#             timeout=10,
#         )
#         response.raise_for_status()
#
#         data = response.json()
#         if not isinstance(data, dict):
#             return None
#
#         return data
#
#     except requests.exceptions.RequestException:
#         return None
#     except ValueError:
#         return None