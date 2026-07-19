from llm_sdk import Small_LLM_Model
import json
import numpy as np
from .parsing import valid
from .engine_core import Engine, TrieNode, argmax
import sys
from pathlib import Path


def main() -> None:
    input_path: str = "data/input/function_calling_tests.json"
    output_path: str = "data/output/function_calling_results.json"
    func_def_path: str = "data/input/functions_definition.json"

    try:
        if len(sys.argv) > 1:
            change = 0
            for arg in sys.argv[1:]:
                if change == 1:
                    func_def_path = arg
                    change = 0
                    continue
                elif change == 2:
                    input_path = arg
                    change = 0
                    continue
                elif change == 3:
                    output_path = arg
                    change = 0
                    continue

                if arg == "--functions_definition":
                    change = 1
                elif arg == "--input":
                    change = 2
                elif arg == "--output":
                    change = 3
                else:
                    raise ValueError(f"'{arg}' is not a valid flag")
    except ValueError as e:
        print(e)
        sys.exit(1)

    model_object = Small_LLM_Model()
    try:
        functions, inputs = valid(func_def_path, input_path)
    except ValueError as e:
        print(e)
        sys.exit(1)
    prompts = [i.prompt for i in inputs]

    path = model_object.get_path_to_vocab_file()
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    dummy_logits = model_object.get_logits_from_input_ids([1])
    vocab_size = len(dummy_logits)

    idx_to_token: list[str | None] = [None] * vocab_size
    idx_to_model_id: list[int] = [0] * vocab_size

    for token_str, token_id in vocab.items():
        if token_id < vocab_size:
            sanitized = token_str.replace("Ċ", "\n").replace("Ġ", " ")
            idx_to_token[token_id] = sanitized
            idx_to_model_id[token_id] = int(token_id)

    allowed_choices = [i.name for i in functions]
    descriptions = []
    for function in functions:
        if function.parameter:
            parameter_names = ", ".join(function.parameter.keys())
        else:
            parameter_names = "none"
        descriptions.append(
            f"function: {function.name} purpose: {function.description} "
            f"args: {parameter_names}"
        )
    float_terminator_mask = np.zeros(vocab_size, dtype=bool)
    int_dot_mask = np.zeros(vocab_size, dtype=bool)

    for idx, token_str in enumerate(idx_to_token):
        if token_str:
            if "," in token_str or "\n" in token_str or "}" in token_str:
                float_terminator_mask[idx] = True
            if "." in token_str:
                int_dot_mask[idx] = True
    i = 1
    data = []
    for prompt in prompts:
        selection_context = (
            f"User request: {prompt}\n\n"
            + "\n".join(descriptions)
        )
        feed = model_object.encode(selection_context)[0].tolist()
        output_str = ""
        prompt = prompt.replace('\\', '\\\\').replace('"', '\\"')
        print(prompt)
        prefix_layout = f'{{\n"prompt": "{prompt}",\n"name": "'

        try:
            print(f"[{i}/{len(prompts)}]'{prompt}' is done")
            i += 1
            obj = json.loads(output_str)
        except json.JSONDecodeError as e:
            print("Invalid output JSON:", e)
            sys.exit(1)
        data.append(obj)

    try:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except TypeError as e:
        print("Data contains a non-JSON type:", e)
    except OSError:
        print("output directory doesn't exist or file doesn't have permission")


if __name__ == "__main__":
    main()
