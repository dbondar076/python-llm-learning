import requests


def fetch_users_safe(url: str) -> list[dict] | None:
    try:
        response = requests.get(url)

        if response.status_code != 200:
            print(f"API error: {response.status_code}")
            return None

        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


if __name__ == "__main__":
    url1 = "https://jsonplaceholder.typicode.com/users"
    url2 = "https://jsonplaceholder.typicode.com/invalid"
    url = "https://wrong-url-123.com"
    users = fetch_users_safe(url)
    print(users)


# import logging
# from typing import Any
# import requests
#
# logger = logging.getLogger(__name__)
#
#
# def fetch_users_safe(url: str) -> list[dict[str, Any]] | None:
#     try:
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#
#         data = response.json()
#
#         if not isinstance(data, list):
#             logger.error("Expected list response from %s", url)
#             return None
#
#         if not all(isinstance(item, dict) for item in data):
#             logger.error("Expected list[dict] response from %s", url)
#             return None
#
#         return data
#
#     except requests.exceptions.Timeout as exc:
#         logger.error("Request timed out for %s: %s", url, exc)
#     except requests.exceptions.RequestException as exc:
#         logger.error("Request failed for %s: %s", url, exc)
#     except ValueError as exc:
#         logger.error("Invalid JSON from %s: %s", url, exc)
#
#     return None