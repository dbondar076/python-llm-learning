import requests

url = "https://jsonplaceholder.typicode.com/users"

def fetch_users() -> list[dict]:
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception("API error")

    return response.json()


def extract_names(users: list[dict]) -> list[str]:
    result = []
    for user in users:
        result.append(user["name"])
    return result


if __name__ == "__main__":
    users = fetch_users()
    names = extract_names(users)
    print(names)