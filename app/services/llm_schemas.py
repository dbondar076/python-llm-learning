def build_analysis_json_schema() -> dict:
    return {
        "type": "json_schema",
        "name": "text_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["question", "statement", "short"],
                },
                "summary": {
                    "type": "string",
                    "maxLength": 80,
                },
            },
            "required": ["category", "summary"],
            "additionalProperties": False,
        },
    }


def build_user_extraction_json_schema() -> dict:
    return {
        "type": "json_schema",
        "name": "user_extract",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1,
                },
                "age": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 120,
                },
            },
            "required": ["name", "age"],
            "additionalProperties": False,
        },
    }