import json
from lesson3 import get_user_label, User


def load_users(file_path: str) -> list[User]:
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_labels(users: list[User]) -> list[str]:
    result: list[str] = []
    for user in users:
        result.append(get_user_label(user))

    return result


def save_labels(labels: list[str], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(labels, file, indent=2)


if __name__ == "__main__":
    users = load_users("users.json")
    labels_list = build_labels(users)
    save_labels(labels_list, "labels.json")