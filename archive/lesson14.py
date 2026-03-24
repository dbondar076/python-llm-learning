from typing import TypedDict


class UserRecord(TypedDict):
    name: str
    age: int


texts = [
    "Name: Dima, Age: 30",
    "Name: Anna, Age: 17",
    "Broken text",
    "Name: Bob, Age: 25",
]


def extract_user_safe(text: str) -> UserRecord | None:
    try:
        name_part, age_part = text.split(",")
        name = name_part.split(":")[1].strip()
        age = int(age_part.split(":")[1].strip())
        return {"name": name, "age": age}
    except (IndexError, ValueError):
        print(f"Failed to parse: {text}")
        return None


def extract_all_safe(texts: list[str]) -> list[UserRecord]:
    result: list[UserRecord] = []

    for text in texts:
        user = extract_user_safe(text)
        if user is not None:
            result.append(user)

    return result


if __name__ == "__main__":
    result = extract_all_safe(texts)
    print(result)