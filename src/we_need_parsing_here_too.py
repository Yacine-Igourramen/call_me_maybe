import json
from pydantic import BaseModel


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


def valid(path_to_funcdef: str, path_to_prompts: str):
    try:
        with open(path_to_funcdef, "r") as f:
            data = json.load(f)
    except OSError:
        raise ValueError("path to function definitions is invalid or doesn't have permision")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    try:
        with open(path_to_prompts, "r") as f:
            prompt = json.load(f)
    except OSError:
        raise ValueError("path to prompts is invalid or doesn't have permision")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
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
