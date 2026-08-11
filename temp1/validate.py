from pydantic import BaseModel


class PromptItem(BaseModel):
    prompt: str


class PromptsList(BaseModel):
    prompts: list[PromptItem]


class ReturnType(BaseModel):
    type: str


class FncItem(BaseModel):
    name: str
    description: str
    parameters: dict[str, ReturnType]
    returns: ReturnType


class FncDefs(BaseModel):
    functions_def: list[FncItem]
