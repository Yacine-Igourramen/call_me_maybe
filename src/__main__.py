import json
from .parsing import valid
from .fsm import FSM
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
    try:
        functions, inputs = valid(func_def_path, input_path)
    except ValueError as e:
        print(e)
        sys.exit(1)
    prompts = [i.prompt for i in inputs]
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
        context = (
            "Select the function that best matches the request.\n\n"
            "--- FUNCTIONS ---\n"
            + "\n".join(descriptions) + "\n\n"
            f"--- USER REQUEST ---\n\"{prompt}\"\n\n"
            "Function Name:"
        )
        output_str = ""
        print(prompt)
        prefix_layout = f'{{"prompt": "{prompt}", "name": "'
        allowed_choices = [i.name for i in functions]
        fsm = FSM(
            allowed_choices=allowed_choices,
            start_string=prefix_layout,
            context=context
        )
        tracker: str = ""
        ch_func: str = ""
        while True:
            allowed_tokens = fsm.allowed_token(fsm.current_state)
            if not allowed_tokens:
                break
            selected_token, text = fsm.mask_logits(allowed_tokens)
            fsm.transition(text)
            tracker += text

            if tracker == prefix_layout:
                fsm.build_func_state()
                tracker = ""
            if tracker in [i.name for i in functions]:
                fsm.build_static_state('", "parameters": {')
                ch_func = tracker
                tracker = ""
            if tracker == '", "parameters": {':
                fsm.build_arg_state(
                    [i.parameter for i in functions if i.name == ch_func][0])
            print(text)
            output_str += text
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
