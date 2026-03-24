from typing import TypedDict

class User(TypedDict):
    name: str
    age: int

def is_adult(user: User) -> bool:
    return user["age"] >= 18

def get_user_label(user: User) -> str:
    status = "adult" if is_adult(user) else "minor"
    return f"{user['name']} ({status})"

