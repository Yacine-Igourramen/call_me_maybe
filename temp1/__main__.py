import builtins
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy
from pydantic import ValidationError
from rich import print

from llm_sdk import Small_LLM_Model
from .parsing import Parser
from .validate import PromptsList
from src.display import Display
from src.fsm import FSM
from src.utils import construct_vocab_pins, format_function, open_json


def initialize_pipeline(parser: Parser) -> tuple[
    Any,
    Any,
    Any,
    Any,
    Any
] | None:
    """
        Handles parser and resource loading in a single block
        to avoid duplicate exception handling.
    """

    try:
        parser.parsing()
        model = Small_LLM_Model()
        input_json = open_json(parser.input)
        valid_prompts = PromptsList(prompts=input_json).prompts
        valid_functions, _ = format_function(parser.fnc_def)

        vocab_path = model.get_path_to_vocab_file()
        vocab_json = open_json(vocab_path)
        vocab_pins = construct_vocab_pins(vocab_json)

        return model, valid_prompts, valid_functions, vocab_json, vocab_pins

    except FileNotFoundError as e:
        print(f"[Error] Required file missing: {e}")
    except (
        json.JSONDecodeError,
        ValidationError,
        KeyError,
        TypeError,
        ValueError
    ) as e:
        print(f"[Error] Data validation or JSON parsing failed: {e}")
    except Exception as e:
        print(f"[Error] Unexpected setup failure: {e}")

    return None


def main() -> None:
    init_res = initialize_pipeline(parse := Parser())
    if not init_res:
        sys.exit(1)

    model, valid_prompts, valid_functions, vocab_json, vocab_pins = init_res
    output_path = Path(parse.output)
    if output_path.parent != Path("."):
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(
                f"[Error] Could not create output directory "
                f"'{output_path.parent}': {e}"
            )
            sys.exit(1)

    try:
        fsm = FSM(valid_functions, vocab_pins, vocab_json)
        fsm.build_state()
    except Exception as e:
        print(f"[Error] Failed to build FSM state: {e}")
        sys.exit(1)

    super_prompt = (
        "<|im_start|>system\n"
        "You are a function-calling assistant.\n"
        "1. Pick the ONE function whose purpose matches the user's "
        "request. Ignore numbers/words that don't fit that purpose.\n"
        "2. Extract ONLY exact values explicitly present in the request "
        "for that function's parameters. Never invent, calculate, or "
        "guess.\n"
        "functions:"
        f"{valid_functions}"
        "<|im_end|>"
        "<|im_start|>user\n"
    )

    try:
        s_prompt_encoded: list[int] = model.encode(super_prompt)[0].tolist()
    except Exception as e:
        print(f"[Error] Prompt encoding failed: {e}")
        sys.exit(1)

    Display()

    start: float = time.time()
    result: list[dict[str, Any]] = []

    for p in valid_prompts:
        fsm.current_state = 0
        output = ""
        pre_p = f"{p.prompt}<|im_end|>\n<|im_start|>assistant\n"

        try:
            p_encoded: list[int] = model.encode(pre_p)[0].tolist()
            full_prompt: list[int] = s_prompt_encoded + p_encoded

            while True:
                logits = model.get_logits_from_input_ids(full_prompt)
                allowed_tokens = fsm.allowed_token(fsm.current_state)
                if not allowed_tokens:
                    break

                masked = fsm.mask_logits(logits, allowed_tokens)
                next_token = int(numpy.argmax(masked))

                full_prompt.append(next_token)
                decoded_token = model.decode([next_token])
                output += decoded_token
                builtins.print(decoded_token, end="", flush=True)
                fsm.transition(decoded_token)
                if fsm.current_state == -1:
                    fsm.current_state = 0
                    break
        except Exception as e:
            print(f"\n[Error] Inference failed for prompt '{p.prompt}': {e}")
            continue

        print("\nAnswer:")
        output_escaped = output.replace("\\", "\\\\")

        try:
            json_format = json.loads(output_escaped)
        except json.JSONDecodeError:
            print(
                f"[Error] Failed to parse model output JSON for prompt: "
                f"{p.prompt!r} -> {output!r}"
            )
            continue

        result.append({"prompt": p.prompt, **json_format})

        try:
            with open(parse.output, "w") as file:
                json.dump(result, file, indent=4)
        except OSError as e:
            print(f"[Error] Failed writing results to {parse.output}: {e}")

    end: float = time.time()
    minutes = int((end - start) // 60)
    seconds = int((end - start) % 60)
    print(f"Call_Me_maybe took: {minutes}m/{seconds}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Iterrupted] program has been interrupted by the user!")
