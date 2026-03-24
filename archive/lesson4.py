from lesson3 import get_user_label, User


users_list: list[User] = [
    {"name": "Dima", "age": 30},
    {"name": "Anna", "age": 17},
    {"name": "Bob", "age": 25},
]


def print_user_labels(users: list[User]) -> None:
    for user in users:
        print(get_user_label(user))


if __name__ == "__main__":
    print_user_labels(users_list)