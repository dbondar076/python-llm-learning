from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


ClassificationLabel = Literal["question", "statement", "short"]


class TextRequest(BaseModel):
    text: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "What is Python?"
            }
        }
    )


class TextAnalysis(BaseModel):
    text: str
    category: ClassificationLabel
    summary: str


class ClassificationResult(BaseModel):
    label: ClassificationLabel


class SummaryResult(BaseModel):
    summary: str


class UserExtractResult(BaseModel):
    name: str = Field(min_length=1)
    age: int = Field(ge=1, le=120)


class SafeTextAnalysis(BaseModel):
    text: str
    category: ClassificationLabel | None
    summary: str
    degraded: bool


# ----------------------------
# RAG
# ----------------------------

class RagChunk(BaseModel):
    doc_id: str
    title: str
    chunk_id: str
    text: str
    score: float


class RagResponse(BaseModel):
    answer: str
    chunks: list[RagChunk]


class RagSearchResponse(BaseModel):
    chunks: list[RagChunk]


class RagSearchRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=20)
    title_filter: str | None = None
    doc_id_filter: str | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "API framework in Python",
                "top_k": 3,
                "title_filter": "FastAPI",
                "doc_id_filter": None,
            }
        }
    )


class RagAnswerRequest(BaseModel):
    question: str
    session_id: str | None = None
    top_k: int = Field(default=3, ge=1, le=20)
    min_score: float = Field(default=0.52, ge=0.0, le=1.0)
    title_filter: str | None = None
    doc_id_filter: str | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What can Python be used for?",
                "session_id": "demo-1",
                "top_k": 3,
                "min_score": 0.52,
                "title_filter": None,
                "doc_id_filter": None,
            }
        }
    )