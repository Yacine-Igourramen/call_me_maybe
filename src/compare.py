from pydantic import BaseModel
from we_need_parsing_here_too import valid


class output(BaseModel):
    prompt: str
    name: str
    args: dict


functions, prompts = valid()

print(functions)
print(prompts)
