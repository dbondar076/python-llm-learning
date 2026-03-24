from dataclasses import dataclass


@dataclass
class User:
    name: str
    age: int


users = [
    User(name="Dima", age=30),
    User(name="Anna", age=17),
    User(name="Bob", age=25),
]


def is_adult(user: User) -> bool:
    return user.age >= 18


def get_user_label(user: User) -> str:
    status = "adult" if is_adult(user) else "minor"
    return f"{user.name} ({status})"


if __name__ == "__main__":
    for user in users:
        print(get_user_label(user))