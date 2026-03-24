import json
from lesson3 import get_user_label


with open("users.json", "r", encoding="utf-8") as file:
    users = json.load(file)

for user in users:
    print(get_user_label(user))