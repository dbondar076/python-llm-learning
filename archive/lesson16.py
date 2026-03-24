from lesson15 import get_api_key


def ask_llm(prompt: str) -> str | None:
    api_key = get_api_key()

    if api_key is None:
        return None

    print(f"Sending prompt: {prompt}")

    return f"Mock response for: {prompt}"

def ask_llm_short(prompt: str) -> str | None:
    return ask_llm(f"{prompt}\n\nAnswer shortly.")


if __name__ == "__main__":
    result = ask_llm("Explain what Python is in one sentence.")
    print(result)