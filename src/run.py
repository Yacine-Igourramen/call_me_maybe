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


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class SglStructuralEngine:
    def __init__(self, allowed_choices: list[str], idx_to_token: list):
        self.idx_to_token = idx_to_token
        self.vocab_size = len(idx_to_token)

        self.trie_root = TrieNode()
        for choice in allowed_choices:
            self._insert_trie(choice)

        self.trie_cache = {}
        self._precompile_trie(self.trie_root)
        self.linear_cache = {}

    def _insert_trie(self, word: str):
        node = self.trie_root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

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

    def set_linear_layout(self, layout_str: str):
        self.linear_cache.clear()
        total_chars = len(layout_str)

        for state in range(total_chars + 1):
            remaining = layout_str[state:]
            valid_ids = []

            if remaining:
                for idx, token_str in enumerate(self.idx_to_token):
                    if token_str is None or token_str == "":
                        continue
                    if remaining.startswith(token_str) and len(token_str) <= len(remaining):
                        valid_ids.append(idx)

            self.linear_cache[state] = np.array(valid_ids, dtype=np.int32)

    def mask_logits_linear(self, logits: np.ndarray, cursor: int) -> np.ndarray:
        allowed_indices = self.linear_cache.get(cursor, np.array([], dtype=np.int32))
        mask = np.full(self.vocab_size, -np.inf, dtype=np.float32)
        if allowed_indices.size > 0:
            mask[allowed_indices] = 0.0
        return logits + mask

    def mask_logits_trie(self, logits: np.ndarray, node: TrieNode) -> np.ndarray:
        allowed_indices = self.trie_cache.get(node, np.array([], dtype=np.int32))
        mask = np.full(self.vocab_size, -np.inf, dtype=np.float32)
        if allowed_indices.size > 0:
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


engine = SglStructuralEngine(allowed_choices, idx_to_token)

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

    print("\n--- Starting Unified Layout & Selection Loop ---")

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

        print(f"[{mode}] Token: {repr(chosen_token_str)}")

        if mode == "PREFIX":
            cursor += len(chosen_token_str)
            if cursor >= len(prefix_layout):
                print("-> Prefix complete. Switching to function selection tree.")
                mode = "TRIE"
                cursor = engine.trie_root

        elif mode == "TRIE":
            selected_fn_name += chosen_token_str
            cursor = engine.advance_trie(cursor, chosen_token_str)

            if cursor.is_end_of_word and not cursor.children:
                print(f"-> Selected function: {selected_fn_name}")

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
                    print(f"-> Ready to generate value for field: {param_keys[current_param_idx]}")
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

    print("\n--- Final Forced Structured JSON Result ---")
    print(output_str)
