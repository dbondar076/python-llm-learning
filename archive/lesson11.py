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