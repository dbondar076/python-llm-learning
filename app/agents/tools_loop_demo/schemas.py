from pydantic import BaseModel, Field


class ToolDecision(BaseModel):
    tool: str
    arguments: dict = Field(default_factory=dict)
    reason: str