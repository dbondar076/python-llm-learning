from pydantic import BaseModel, Field


class AgentAction(BaseModel):
    type: str
    tool_name: str
    arguments: dict = Field(default_factory=dict)
    reason: str