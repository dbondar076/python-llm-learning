from typing import TypedDict


class UserRecord(TypedDict):
    name: str
    age: int


texts = [
    "Name: Dima, Age: 30",
    "Name: Anna, Age: 17",
    "Name: Bob, Age: 25",
]


def extract_user(text: str) -> UserRecord:
    name_part, age_part = text.split(",")
    name = name_part.split(":")[1].strip()
    age = int(age_part.split(":")[1].strip())

    return {"name": name, "age": age}


def extract_all(texts: list[str]) -> list[UserRecord]:
    return [extract_user(text) for text in texts]


if __name__ == "__main__":
    result = extract_all(texts)
    print(result)


# from typing import TypedDict
#
#
# class UserRecord(TypedDict):
#     name: str
#     age: int
#
#
# def extract_user(text: str) -> UserRecord | None:
#     try:
#         parts = text.split(",")
#
#         if len(parts) != 2:
#             return None
#
#         name_part, age_part = parts
#
#         name_items = name_part.split(":", 1)
#         age_items = age_part.split(":", 1)
#
#         if len(name_items) != 2 or len(age_items) != 2:
#             return None
#
#         name = name_items[1].strip()
#         age = int(age_items[1].strip())
#
#         return {"name": name, "age": age}
#     except ValueError:
#         return None
#
#
# def extract_all(texts: list[str]) -> list[UserRecord]:
#     result = []
#
#     for text in texts:
#         user = extract_user(text)
#         if user is not None:
#             result.append(user)
#
#     return result