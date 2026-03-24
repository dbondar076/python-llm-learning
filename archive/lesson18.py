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