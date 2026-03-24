from pydantic import BaseModel, ValidationError


class User(BaseModel):
    name: str
    age: int


raw_users = [
    {"name": "Dima", "age": 30},
    {"name": "Anna", "age": 17},
    {"name": "Bob", "age": 25},
    {"name": "Kate", "age": "hello"}
]


def parse_users_safe(data: list[dict]) -> list[User]:
    result: list[User] = []

    for item in data:
        try:
            user = User(**item)
            result.append(user)
        except ValidationError as e:
            print(f"Failed to parse user: {item}")
            print(e.errors())

    return result


if __name__ == "__main__":
    users = parse_users_safe(raw_users)
    print(users)

    for user in users:
        print(user.name, user.age)