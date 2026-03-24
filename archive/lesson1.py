from lesson2 import get_user_label

users = [
    {"name": "Dima", "age": 30},
    {"name": "Anna", "age": 17},
    {"name": "Bob", "age": 25}
]

for user in users:
    print(
        f"User {user['name']} is {user['age']} years old ({'adult' if user['age'] >= 18 else 'minor'})"
    )

names = [user["name"] for user in users]
print(names)

for user in users:
    print(get_user_label(user))
