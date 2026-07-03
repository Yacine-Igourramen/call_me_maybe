from llm_sdk import Small_LLM_Model
import json
import numpy as np
from we_need_parsing_here_too import valid


model_object = Small_LLM_Model()

functions, inputs = valid()
prompts = [i.prompt for i in inputs]

path = model_object.get_path_to_vocab_file()
with open(path, "r", encoding="utf-8") as f:
    vocab = json.load(f)

dummy_logits = model_object.get_logits_from_input_ids([1])
vocab_size = len(dummy_logits)

idx_to_token = [None] * vocab_size
for token_str, idx in vocab.items():
    if idx < vocab_size:
        sanitized = token_str
        sanitized = sanitized.replace("Ċ", "\n")
        sanitized = sanitized.replace("Ġ", " ")
        idx_to_token[idx] = sanitized

allowed_choices = [i.name for i in functions]
descriptions = []
for function in functions:
    if function.parameter:
        parameter_names = ", ".join(function.parameter.keys())
    else:
        parameter_names = "none"

    if function.returns:
        return_type = function.returns.get("type")
    else:
        return_type = "unknown"

    descriptions.append(
        f"function: {function.name} description: {function.description} "
        f"parameters: {parameter_names} returns: {return_type}"
    )


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class SglFusedEngine:
    def __init__(
        self,
        schema_template: str,
        allowed_choices: list[str],
        idx_to_token: list,
    ):
        self.idx_to_token = idx_to_token
        self.vocab_size = len(idx_to_token)
        self.slot_marker = "<LIST_CHOICE>"

        if self.slot_marker not in schema_template:
            raise ValueError(
                f"Template must contain '{self.slot_marker}' to know where to "
                "enforce the list."
            )

        self.prefix_layout, self.suffix_layout = schema_template.split(
            self.slot_marker
        )

        self.trie_root = TrieNode()
        for choice in allowed_choices:
            self._insert_trie(choice)

        self.linear_cache = {}
        self.trie_cache = {}
        print("Pre-compiling Fused State Engine Maps... (This runs ONCE)")
        self._precompile_linear(self.prefix_layout, "prefix")
        self._precompile_trie(self.trie_root)

    def _insert_trie(self, word: str):
        node = self.trie_root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def _precompile_linear(self, layout_str: str, key_prefix: str):
        total_chars = len(layout_str)
        for state in range(total_chars + 1):
            remaining = layout_str[state:]
            valid_ids = []
            if remaining:
                for idx, token_str in enumerate(self.idx_to_token):
                    if token_str is None or token_str == "":
                        continue
                    if remaining.startswith(token_str):
                        valid_ids.append(idx)
                        continue
                    if token_str.startswith(remaining) and (
                        state + len(remaining) == total_chars
                    ):
                        valid_ids.append(idx)
            self.linear_cache[f"{key_prefix}_{state}"] = np.array(
                valid_ids,
                dtype=np.int32,
            )

    def _precompile_trie(self, node: TrieNode):
        valid_ids = []
        for idx, token_str in enumerate(self.idx_to_token):
            if token_str is None or token_str == "":
                continue

            curr = node
            possible = True
            for char in token_str:
                if char not in curr.children:
                    possible = False
                    break
                curr = curr.children[char]
            if possible:
                valid_ids.append(idx)

        self.trie_cache[node] = np.array(valid_ids, dtype=np.int32)
        for child in node.children.values():
            self._precompile_trie(child)

    def mask_logits(self, logits: np.ndarray, mode: str, cursor) -> np.ndarray:
        if mode in ("prefix", "args"):
            allowed_indices = self.linear_cache.get(
                f"{mode}_{cursor}",
                np.array([], dtype=np.int32),
            )
        elif mode == "trie":
            allowed_indices = self.trie_cache.get(
                cursor,
                np.array([], dtype=np.int32),
            )
        else:
            allowed_indices = np.array([], dtype=np.int32)

        mask = np.full(self.vocab_size, -np.inf, dtype=np.float32)
        if allowed_indices is not None and allowed_indices.size > 0:
            mask[allowed_indices] = 0.0
        return logits + mask

    def advance_trie(self, node: TrieNode, token_str: str) -> TrieNode:
        curr = node
        for char in token_str:
            curr = curr.children.get(char)
            if curr is None:
                return None
        return curr


def argmax(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def build_args_layout(function) -> str:
    if not function.parameter:
        return "{\n}"

    lines = []
    for field_name in function.parameter.keys():
        lines.append(f'"{field_name}": null')

    return "{\n" + ",\n".join(lines) + "\n}"


for prompt in prompts:
    escaped_prompt = prompt.replace('"', '\\"')

    exp_output = (
        f'{{\n"prompt": "{escaped_prompt}",\n'
        '"fn_name": "<LIST_CHOICE>",\n'
        '"args": <ARGS_CHOICE>\n}}'
    )

    prefix_layout, rest_layout = exp_output.split("<LIST_CHOICE>")
    choice_layout, suffix_layout = rest_layout.split("<ARGS_CHOICE>")

    engine = SglFusedEngine(
        prefix_layout + "<LIST_CHOICE>" + choice_layout,
        allowed_choices,
        idx_to_token,
    )

    selection_context = (
        "Choose the best function for the user request.\n"
        f"User request: {prompt}\n\n"
        "Available functions:\n"
        + "\n".join(descriptions)
    )

    feed = model_object.encode(selection_context)[0].tolist()
    output_str = ""

    mode = "prefix"
    cursor = 0
    args_layout = ""

    print("\n--- Starting Unified Layout & Selection Loop ---")

    while True:
        raw_logits = np.array(
            model_object.get_logits_from_input_ids(feed),
            dtype=np.float32,
        )

        masked_logits = engine.mask_logits(raw_logits, mode, cursor)
        next_token_idx = argmax(masked_logits)

        if masked_logits[next_token_idx] == -np.inf:
            print(
                f"\nBottleneck encountered in [{mode.upper()}] "
                f"at index {cursor}. Forcing safety exit."
            )
            break

        chosen_token_str = idx_to_token[next_token_idx]
        feed.append(next_token_idx)
        output_str += chosen_token_str

        if mode == "prefix":
            cursor += len(chosen_token_str)
            if cursor >= len(engine.prefix_layout):
                print("-> Layout prefix complete.")
                mode = "trie"
                cursor = engine.trie_root
        elif mode == "trie":
            next_node = engine.advance_trie(cursor, chosen_token_str)
            if next_node is not None:
                cursor = next_node

            if cursor.is_end_of_word and (
                not cursor.children or next_node is None
            ):
                print("-> Exact match hit.")

                selected_function_obj = None
                for function in functions:
                    if function.name in output_str:
                        selected_function_obj = function
                        break

                if selected_function_obj is None:
                    raise ValueError(
                        "Could not determine selected function name."
                    )

                args_layout = (
                    choice_layout
                    + build_args_layout(selected_function_obj)
                    + suffix_layout
                )
                engine._precompile_linear(args_layout, "args")
                mode = "args"
                cursor = 0
        elif mode == "args":
            cursor += len(chosen_token_str)
            if cursor >= len(args_layout):
                break

    print("\n--- Final Forced Structured JSON Result ---")
    print(output_str)
