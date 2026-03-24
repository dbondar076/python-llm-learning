import requests

HTTPBIN_POST_URL = "https://httpbin.org/post"


def send_prompt(prompt: str) -> dict | None:
    try:
        response = requests.post(
            HTTPBIN_POST_URL,
            json={
                "prompt": prompt,
                "mode": "short",
            },
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
    result = send_prompt("Explain what Python is in one sentence.")
    if result is not None:
        print(result["json"])