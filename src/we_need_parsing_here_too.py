import json
from pydantic import BaseModel, create_model


class validator(BaseModel):
    name: str
    description: str
    parameter: dict | None
    returns: dict | None


class prompt_valid(BaseModel):
    prompt: str


def args_parser(args: dict):
    type_map = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
    }
    fields = {}

    for field_name, info in args.items():
        python_type = type_map[info["type"]]
        fields[field_name] = python_type

    return fields


def valid():
    with open("../data/input/functions_definition.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    with open("../data/input/function_calling_tests.json", "r", encoding="utf-8") as f:
        prompt = json.load(f)

    functions: list[validator] = []
    prompts: list[prompt_valid] = []

    try:
        for i in data:
            obj = validator(name=i["name"],
                            description=i["description"],
                            parameter=args_parser(i["parameters"]),
                            returns=i["returns"])
            functions.append(obj)
        for i in prompt:
            obj = prompt_valid(prompt=i["prompt"])
            prompts.append(obj)
    except Exception:
        print("functions_definition must include a name, description, parameters and return")

    return functions, prompts


if __name__ == "__main__":
    valid()
