import json


def load_json_file(file_path: str) -> list[dict] | None:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Invalid JSON: {file_path}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None


if __name__ == "__main__":
    data = load_json_file("broken.json")
    print(data)