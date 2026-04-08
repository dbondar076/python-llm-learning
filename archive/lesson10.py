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


# import json
# import logging
# from typing import Any
#
# logger = logging.getLogger(__name__)
#
#
# def load_json_file(file_path: str) -> list[dict[str, Any]] | None:
#     try:
#         with open(file_path, "r", encoding="utf-8") as file:
#             data = json.load(file)
#
#         if not isinstance(data, list):
#             logger.error("Expected a list in JSON file: %s", file_path)
#             return None
#
#         if not all(isinstance(item, dict) for item in data):
#             logger.error("Expected list[dict] in JSON file: %s", file_path)
#             return None
#
#         return data
#
#     except FileNotFoundError:
#         logger.error("File not found: %s", file_path)
#     except json.JSONDecodeError as exc:
#         logger.error("Invalid JSON in %s: %s", file_path, exc)
#     except OSError as exc:
#         logger.error("OS error while reading %s: %s", file_path, exc)
#
#     return None