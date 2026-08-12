from llm_sdk import Small_LLM_Model
import json
from .parsing import valid
from .engine_core import (
    set_linear_layout,
    get_static_token_idx,
    mask_logits_vectorized,
    argmax,
)
from .fsm import ArgumentFSM
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

    dummy_logits = model_object.get_logits_from_input_ids([1])
    vocab_size = len(dummy_logits)

    idx_to_token = [None] * vocab_size
    id_to_raw_token = [None] * vocab_size

    for token_str, token_id in vocab.items():
        sanitized = token_str.replace("Ċ", "\n").replace("Ġ", " ")
        idx_to_token[token_id] = sanitized
        id_to_raw_token[token_id] = token_str

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
    j = 1
    data = []
    for prompt in prompts:
        selection_context = (
            "Select the function that best matches the request.\n\n"
            "--- FUNCTIONS ---\n"
            + "\n".join(descriptions) + "\n\n"
            f"--- USER REQUEST ---\n\"{prompt}\"\n\n"
            "Function Name:"
        )
        feed = model_object.encode(selection_context)[0].tolist()
        output_str = ""
        prompt = prompt.replace('\\', '\\\\').replace('"', '\\"')
        print(prompt)
        prefix_layout = f'{{\n"prompt": "{prompt}",\n"name": "'
        prefix_tokens = set_linear_layout(prefix_layout, idx_to_token)
        mode = "prefix"
        cursor = 0
        chosen_func = ""
        while True:
            if mode == "prefix":
                if len(output_str) == len(prefix_layout):
                    mode = "choices"
                    continue
                selected_token = get_static_token_idx(cursor, prefix_tokens)
            elif mode == "choices":
                logits = model_object.get_logits_from_input_ids(feed)
                logits = mask_logits_vectorized(
                    logits, chosen_func, allowed_choices, idx_to_token
                )
                if logits is None:
                    # transition to args-generation mode and build FSM
                    mode = "args"
                    # determine the selected function object
                    target_function = None
                    for fun in functions:
                        if fun.name == chosen_func:
                            target_function = fun
                            break

                    # prepare formatted parameter schema for FSM
                    reverse_type_map = {
                        str: "string",
                        float: "number",
                        int: "integer",
                        bool: "boolean",
                    }
                    formatted_schema = {}
                    if target_function and target_function.parameter:
                        for pname, ptype in target_function.parameter.items():
                            formatted_schema[pname] = {
                                "type": reverse_type_map.get(ptype, "string")}

                    # build vocab pins and all_vocab from raw vocab mapping
                    vocab_pins = {}
                    all_vocab = vocab.copy()
                    for tstr, tid in vocab.items():
                        if not tstr:
                            continue
                        key = tstr[0]
                        vocab_pins.setdefault(key, {})[tstr] = tid
                    start_string = '", "parameters": {'
                    output_str += start_string
                    feed.extend(model_object.encode(start_string)[0].tolist())
                    fsm = ArgumentFSM(
                        parameters=formatted_schema,
                        vocab_pins=vocab_pins,
                        all_vocab=all_vocab,
                        start_string=start_string
                    )
                    fsm.current_state = len(fsm.tr_token(fsm.start_string))
                    continue
                selected_token = argmax(logits)
                chosen_func += idx_to_token[selected_token]
            elif mode == "args":
                logits = model_object.get_logits_from_input_ids(feed)
                # get allowed token ids from FSM and mask logits accordingly
                allowed_tokens = fsm.allowed_token(fsm.current_state)
                if not allowed_tokens:

                    break
                logits = fsm.mask_logits(logits, allowed_tokens)
                selected_token = argmax(logits)
                # advance FSM with the chosen token string in raw vocab form
                raw_token = id_to_raw_token[selected_token]
                # debug: show transition input and state change
                fsm.transition(raw_token)

            print(idx_to_token[selected_token])
            cursor += len(idx_to_token[selected_token])
            feed.append(selected_token)
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
