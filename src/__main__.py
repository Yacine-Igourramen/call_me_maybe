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
    engine = Engine(allowed_choices, idx_to_token)
    i = 1
    data = []
    for prompt in prompts:
        selection_context = (
            "Analyze the user request carefully. Choose the correct function "
            "name and fill in the parameters accurately.\n"
            f"User request: {prompt}\n\n"
            "Available functions definitions pick correct ones:\n"
            + "\n".join(descriptions)
        )
        feed = model_object.encode(selection_context)[0].tolist()
        output_str = ""
        prompt = prompt.replace('\\', '\\\\').replace('"', '\\"')
        print(prompt)
        prefix_layout = f'{{\n"prompt": "{prompt}",\n"name": "'
        engine.set_linear_layout(prefix_layout)

        mode = "PREFIX"
        linear_cursor = 0
        trie_cursor: TrieNode = None

        selected_fn_name = ""
        param_keys: list[str] = []
        current_param_idx = 0
        current_static_layout = ""

        dynamic_value_buffer = ""
        selected_func = None

        while True:
            raw_logits = np.array(
                model_object.get_logits_from_input_ids(feed), dtype=np.float32
            )

            if mode == "PREFIX":
                masked_logits = engine.mask_logits_linear(
                    raw_logits, linear_cursor
                )
            elif mode == "TRIE":
                masked_logits = engine.mask_logits_trie(raw_logits,
                                                        trie_cursor)
            elif mode == "ARGS_STATIC":
                masked_logits = engine.mask_logits_linear(
                    raw_logits, linear_cursor
                )
            elif mode == "ARGS_DYNAMIC":
                masked_logits = raw_logits

                is_float_type = False
                is_int_type = False
                if selected_func and selected_func.parameter:
                    current_param = param_keys[current_param_idx]
                    param_type = (
                        selected_func.parameter.get(current_param))
                    is_float_type = (param_type == float)
                    is_int_type = (param_type == int)

                if is_float_type and "." not in dynamic_value_buffer:
                    masked_logits[float_terminator_mask] = -np.inf

                # Replace the old loop with this:
                if is_int_type:
                    masked_logits[int_dot_mask] = -np.inf

            next_token_idx = argmax(masked_logits)
            chosen_token_str = idx_to_token[next_token_idx]
            print(f"the chosen token is: {chosen_token_str}")
            feed.append(idx_to_model_id[next_token_idx])
            output_str += chosen_token_str

            if mode == "PREFIX":
                linear_cursor += len(chosen_token_str)
                if linear_cursor >= len(prefix_layout):
                    mode = "TRIE"
                    trie_cursor = engine.trie_root

            elif mode == "TRIE":
                selected_fn_name += chosen_token_str
                next_trie_cursor = engine.advance_trie(
                    trie_cursor, chosen_token_str
                )
                if next_trie_cursor is None:
                    raise ValueError("invalid trie transition")
                trie_cursor = next_trie_cursor

                if trie_cursor.is_end_of_word and not trie_cursor.children:
                    selected_func = next(
                        (f for f in functions if f.name == selected_fn_name),
                        None,
                    )

                    if (
                        selected_func
                        and selected_func.parameter
                    ):
                        param_keys = list(
                            selected_func.parameter.keys()
                        )
                        current_param_idx = 0
                        current_param = param_keys[current_param_idx]

                        param_type = (
                            selected_func.parameter.get(current_param))
                        if param_type in (int, float):
                            current_static_layout = (
                                f'",\n"parameters": {{\n"{current_param}": ')
                        else:
                            current_static_layout = (
                                f'",\n"parameters": {{\n"{current_param}": "')

                        engine.set_linear_layout(current_static_layout)
                        mode = "ARGS_STATIC"
                        linear_cursor = 0
                    else:
                        current_static_layout = '",\n"parameters": {}\n}'
                        engine.set_linear_layout(current_static_layout)
                        mode = "ARGS_STATIC"
                        linear_cursor = 0

            elif mode == "ARGS_STATIC":
                linear_cursor += len(chosen_token_str)
                if linear_cursor >= len(current_static_layout):
                    if param_keys and current_param_idx < len(param_keys):
                        mode = "ARGS_DYNAMIC"
                        dynamic_value_buffer = ""
                    else:
                        break

            elif mode == "ARGS_DYNAMIC":
                is_numeric = False
                is_float_type = False
                if selected_func and selected_func.parameter:
                    prev_param_name = param_keys[current_param_idx]
                    param_type = (
                        selected_func.parameter.get(prev_param_name))
                    is_numeric = param_type in (int, float)
                    is_float_type = (param_type == float)

                is_terminator = False
                if is_numeric:
                    is_terminator = (
                        "," in chosen_token_str
                        or "\n" in chosen_token_str
                        or "}" in chosen_token_str
                    )
                else:
                    is_terminator = ('"' in chosen_token_str)

                if not is_terminator:
                    dynamic_value_buffer += chosen_token_str
                else:
                    current_param_idx += 1

                    if is_numeric:
                        terminator_chars = [",", "\n", "}"]
                        idx = len(chosen_token_str)
                        for char in terminator_chars:
                            if char in chosen_token_str:
                                idx = (
                                    min(idx, chosen_token_str.index(char)))

                        clean_token = chosen_token_str[:idx].strip()

                        if is_float_type and (
                            "." not in dynamic_value_buffer
                            and "." not in clean_token
                        ):
                            clean_token += ".0"

                        output_str = (
                            output_str[:-len(chosen_token_str)] + clean_token)
                    else:
                        clean_token = chosen_token_str.split('"')[0] + '"'
                        output_str = (
                            output_str[:-len(chosen_token_str)] + clean_token)

                    if current_param_idx < len(param_keys) and selected_func:
                        next_param = param_keys[current_param_idx]
                        next_param_type = (
                            selected_func.parameter.get(next_param))

                        if next_param_type in (int, float):
                            current_static_layout = f',\n"{next_param}": '
                        else:
                            current_static_layout = f',\n"{next_param}": "'

                        engine.set_linear_layout(current_static_layout)
                        mode = "ARGS_STATIC"
                        linear_cursor = 0
                    else:
                        current_static_layout = "\n}\n}"
                        engine.set_linear_layout(current_static_layout)
                        mode = "ARGS_STATIC"
                        linear_cursor = 0
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
