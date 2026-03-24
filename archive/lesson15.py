import os

from dotenv import load_dotenv


def get_api_key() -> str | None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("OPENAI_API_KEY is missing")
        return None

    return api_key


if __name__ == "__main__":
    api_key = get_api_key()
    print(api_key)