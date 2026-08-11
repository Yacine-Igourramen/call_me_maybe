import json
from typing import Any
from .validate import FncDefs


def open_json(file: str) -> Any:
    with open(file) as f:
        return json.load(f)


def format_function(fnc_path: str) -> tuple[Any, str]:
    fnc_json = open_json(fnc_path)
    functions = FncDefs(functions_def=fnc_json).functions_def
    function_format = "\n".join(
        f"{func.name}("
        f"{', '.join(f'{k}: {v.type}' for k, v in func.parameters.items())}"
        f")- {func.description}"
        for func in functions
    )
    return (functions, function_format)


def construct_vocab_pins(vocab_json: dict[str, Any]) -> dict[str, Any]:
    vocab_pins: dict[str, Any] = {}
    for key in vocab_json:
        vocab_pins.setdefault(key[0], {}).update(
            {key: vocab_json[key]}
        )
    return vocab_pins
