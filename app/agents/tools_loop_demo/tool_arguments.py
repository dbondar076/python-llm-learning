from pydantic import BaseModel, Field


class CalculatorArguments(BaseModel):
    expression: str = Field(..., min_length=1)


class SearchChunksArguments(BaseModel):
    # пока пусто, но структура есть
    pass


class ListDocsArguments(BaseModel):
    # тоже пусто
    pass