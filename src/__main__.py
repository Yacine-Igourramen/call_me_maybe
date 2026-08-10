from llm_sdk import Small_LLM_Model
import json
import numpy as np
from .parsing import valid
from .engine_core import set_linear_layout, get_static_token_idx, mask_logits_vectorized, argmax, pre_compile_args, mask_dynamic_args, encode
import sys
from pathlib import Path

token_limit = 50


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
    print(len(vocab))

    dummy_logits = model_object.get_logits_from_input_ids([1])
    vocab_size = len(dummy_logits)
    print(vocab_size)

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
    j = 1
    data = []
    vocab_strings = np.array([t if t else "" for t in idx_to_token], dtype=object)
    for prompt in prompts:
        selection_context = (
            "Select the function that best matches the request.\n\n"
            "--- FUNCTIONS ---\n"
            + "\n".join(descriptions) + "\n\n"
            f"--- USER REQUEST ---\n\"{prompt}\"\n\n"
            "Function Name:"
        )
        feed = encode(selection_context)
        output_str = ""
        prompt = prompt.replace('\\', '\\\\').replace('"', '\\"')
        print(prompt)
        prefix_layout = f'{{\n"prompt": "{prompt}",\n"name": "'
        prefix_tokens = set_linear_layout(prefix_layout, idx_to_token)
        mode = "prefix"
        cursor = 0
        chosen_func = ""
        the_arg = ""
        i = 0
        b = 0
        done = False
        while True:
            if mode == "prefix":
                if len(output_str) == len(prefix_layout):
                    mode = "choices"
                    continue
                selected_token = get_static_token_idx(cursor, prefix_tokens)
            elif mode == "choices":
                logits = model_object.get_logits_from_input_ids(feed)
                logits = mask_logits_vectorized(
                    logits, chosen_func, allowed_choices, vocab_strings
                )
                if logits is None:
                    mode = "args"
                    args_layout_str, prefix_tokens, param_list = pre_compile_args(chosen_func, functions, idx_to_token)
                    cursor = 0
                    continue
                selected_token = argmax(logits)
                chosen_func += idx_to_token[selected_token]
            elif mode == "args":
                if i == len(prefix_tokens):
                    break
                selected_token = get_static_token_idx(cursor, prefix_tokens[i])
                if cursor >= len(args_layout_str[i]):
                    i += 1
                    cursor = 0
                    if b < len(param_list):
                        mode = "dynamic_args"
                        the_arg = ""
                    continue
            elif mode == "dynamic_args":
                logits = model_object.get_logits_from_input_ids(feed)
                logits = mask_dynamic_args(logits, param_list[b], idx_to_token, the_arg)
                selected_token = argmax(logits)
                token_str = idx_to_token[selected_token]
                end = ['}', ',', '\n']
                if param_list[b] == str:
                    end = ['"']
                for char in end:
                    if char in token_str and '\\"' not in token_str:

                        print(f"end character is {token_str}")
                        token_str = token_str[:token_str.find(char)]
                        selected = encode(token_str)
                        for token in selected:
                            print(idx_to_token[token])
                            feed.append(idx_to_model_id[token])
                            output_str += idx_to_token[token]
                        done = True
                if done:
                    mode = "args"
                    done = False
                    the_arg = ""
                    b += 1
                    cursor = 0
                    continue
                the_arg += token_str

            print(idx_to_token[selected_token])
            cursor += len(idx_to_token[selected_token])
            feed.append(idx_to_model_id[selected_token])
            output_str += idx_to_token[selected_token]
        try:
            print(f"[{j}/{len(prompts)}]'{prompt}' is done")
            j += 1
            print(output_str)
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
