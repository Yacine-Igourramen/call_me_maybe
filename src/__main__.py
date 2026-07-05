from llm_sdk import Small_LLM_Model
import json
import numpy as np
from we_need_parsing_here_too import valid
from run import SglStructuralEngine, argmax
import sys

input_path: str = "data/input/function_calling_tests.json"
output_path: str = "data/output/function_calling_results.json"
func_def_path: str = "data/input/functions_definition.json"

try:
    if len(sys.argv) > 2:
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
                raise ValueError("not a valid flag")
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

idx_to_token = [None] * vocab_size
idx_to_model_id = [0] * vocab_size

for token_str, token_id in vocab.items():
    if token_id < vocab_size:
        sanitized = token_str.replace("Ċ", "\n").replace("Ġ", " ")
        idx_to_token[token_id] = sanitized
        idx_to_model_id[token_id] = int(token_id)

allowed_choices = [i.name for i in functions]
descriptions = []
for function in functions:
    parameter_names = ", ".join(function.parameter.keys()) if function.parameter else "none"
    descriptions.append(f"function: {function.name} description: {function.description} parameters: {parameter_names}")

engine = SglStructuralEngine(allowed_choices, idx_to_token)
data = []
for prompt in prompts:
    selection_context = (
        "Analyze the user request carefully. Choose the correct function name and fill in the parameters accurately.\n"
        f"User request: {prompt}\n\n"
        "Available functions definitions:\n"
        + "\n".join(descriptions)
        + "\n\nResponse:\n"
    )

    feed = model_object.encode(selection_context)[0].tolist()
    output_str = ""
    prompt = prompt.replace('"', '\\"')
    prefix_layout = f'{{\n"prompt": "{prompt}",\n"fn_name": "'
    engine.set_linear_layout(prefix_layout)

    mode = "PREFIX"
    cursor = 0

    selected_fn_name = ""
    param_keys = []
    current_param_idx = 0
    current_static_layout = ""

    while True:
        raw_logits = np.array(model_object.get_logits_from_input_ids(feed), dtype=np.float32)

        if mode == "PREFIX":
            masked_logits = engine.mask_logits_linear(raw_logits, cursor)
        elif mode == "TRIE":
            masked_logits = engine.mask_logits_trie(raw_logits, cursor)
        elif mode == "ARGS_STATIC":
            masked_logits = engine.mask_logits_linear(raw_logits, cursor)
        elif mode == "ARGS_DYNAMIC":
            masked_logits = raw_logits

        next_token_idx = argmax(masked_logits)

        if masked_logits[next_token_idx] == -np.inf:
            print(f"\nBottleneck encountered in [{mode}] at index {cursor}. Forcing safety exit.")
            break

        chosen_token_str = idx_to_token[next_token_idx]
        feed.append(idx_to_model_id[next_token_idx])
        output_str += chosen_token_str

        if mode == "PREFIX":
            cursor += len(chosen_token_str)
            if cursor >= len(prefix_layout):
                mode = "TRIE"
                cursor = engine.trie_root

        elif mode == "TRIE":
            selected_fn_name += chosen_token_str
            cursor = engine.advance_trie(cursor, chosen_token_str)

            if cursor.is_end_of_word and not cursor.children:

                selected_function_obj = next((f for f in functions if f.name == selected_fn_name), None)

                if selected_function_obj and selected_function_obj.parameter:
                    param_keys = list(selected_function_obj.parameter.keys())
                    current_param_idx = 0
                    current_static_layout = f'",\n"args": {{\n"{param_keys[current_param_idx]}": "'
                    engine.set_linear_layout(current_static_layout)
                    mode = "ARGS_STATIC"
                    cursor = 0
                else:
                    current_static_layout = '",\n"args": {}\n}'
                    engine.set_linear_layout(current_static_layout)
                    mode = "ARGS_STATIC"
                    cursor = 0

        elif mode == "ARGS_STATIC":
            cursor += len(chosen_token_str)
            if cursor >= len(current_static_layout):
                if param_keys and current_param_idx < len(param_keys):
                    mode = "ARGS_DYNAMIC"
                else:
                    break

        elif mode == "ARGS_DYNAMIC":
            if '"' in chosen_token_str:
                current_param_idx += 1
                if current_param_idx < len(param_keys):
                    # Localized slice: strip away trailing tokens after the quote mark for this parameter value
                    clean_token = chosen_token_str.split('"')[0] + '"'

                    # Remove the raw un-sliced token from the end of the accumulator and add the clean one
                    output_str = output_str[:-len(chosen_token_str)] + clean_token

                    current_static_layout = f',\n"{param_keys[current_param_idx]}": "'
                    engine.set_linear_layout(current_static_layout)
                    mode = "ARGS_STATIC"
                    cursor = 0
                else:
                    # Final slice: strip trailing tokens from this last parameter value, then append the structural brackets
                    clean_token = chosen_token_str.split('"')[0]
                    output_str = output_str[:-len(chosen_token_str)] + clean_token + '"\n}\n}'
                    break
    try:
        obj = json.loads(output_str)
    except json.JSONDecodeError as e:
        print("Invalid JSON:", e)
        sys.exit(1)
    data.append(obj)

try:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
except TypeError as e:
    print("Data contains a non-JSON type:", e)
except OSError:
    print("output directory doesn't exist or file doesn't have permision")
