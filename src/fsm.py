from typing import Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class ArgumentFSM(BaseModel):
    """
    Finite-state machine for constrained JSON parameter generation.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=False)

    parameters: dict[str, Any]
    vocab_pins: dict[str, dict[str, int]]
    all_vocab: dict[str, int]
    start_string: str = '", "parameters": {'

    state: int = Field(default=0)
    current_state: int = Field(default=0)

    prefix_tree: dict[int, dict[Optional[str], int]] = Field(default_factory=dict)
    cache: dict[int, list[int]] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        self.build_state()

    def tr_token(self, string: str) -> str:
        return string.replace(" ", "Ġ")

    def build_state(self) -> None:
        start = self.tr_token(self.start_string)
        for c in start:
            self.prefix_tree.setdefault(self.state, {})[c] = self.state + 1
            self.state += 1

        num_params = len(self.parameters)
        if num_params == 0:
            self.prefix_tree.setdefault(self.state, {})["}"] = self.state + 1
            self.prefix_tree.setdefault(self.state + 1, {})["}"] = self.state + 2
            self.state += 2
            return

        for index, (key, param_obj) in enumerate(self.parameters.items(), 1):
            prefix = f' "{key}": ' if index > 1 else f'"{key}": '

            for c in self.tr_token(prefix):
                self.prefix_tree.setdefault(self.state, {})[c] = self.state + 1
                self.state += 1

            exit_states = self.build_type_state(param_obj)
            is_last_param = index == num_params

            if is_last_param:
                brace_1 = self.state + 1
                for exit_s in exit_states:
                    self.prefix_tree.setdefault(exit_s, {})["}"] = brace_1

                brace_2 = brace_1 + 1
                self.prefix_tree.setdefault(brace_1, {})["}"] = brace_2
                self.state = brace_2
            else:
                next_s = self.state + 1
                for exit_s in exit_states:
                    self.prefix_tree.setdefault(exit_s, {})[","] = next_s
                self.state = next_s

    def build_type_state(self, param_obj: Any) -> list[int]:
        if hasattr(param_obj, "type"):
            param_type = param_obj.type
        elif isinstance(param_obj, dict):
            param_type = param_obj.get("type", "string")
        else:
            param_type = "string"

        if param_type == "integer":
            return self.build_int_state()
        elif param_type == "number":
            return self.build_number_state()
        else:
            return self.build_string_state()

    def build_string_state(self) -> list[int]:
        entry_s = self.state

        self.prefix_tree.setdefault(entry_s, {})['"'] = entry_s + 1
        self.state += 1

        self.prefix_tree.setdefault(self.state, {})[None] = self.state
        self.prefix_tree[self.state]['"'] = self.state + 1
        self.state += 1

        return [self.state]

    def build_number_state(self) -> list[int]:
        base_s = self.state
        f_nums = "-0123456789"
        for c in f_nums:
            if c == "-":
                self.prefix_tree.setdefault(base_s, {})[c] = base_s + 1
            elif c == "0":
                self.prefix_tree.setdefault(base_s, {})[c] = base_s + 2
            else:
                self.prefix_tree.setdefault(base_s, {})[c] = base_s + 3

        for c in "0123456789":
            if c == "0":
                self.prefix_tree.setdefault(base_s + 1, {})[c] = base_s + 2
            else:
                self.prefix_tree.setdefault(base_s + 1, {})[c] = base_s + 3

        self.prefix_tree.setdefault(base_s + 2, {})["."] = base_s + 4
        self.prefix_tree.setdefault(base_s + 3, {})["."] = base_s + 4

        for c in "0123456789":
            self.prefix_tree.setdefault(base_s + 3, {})[c] = base_s + 3

        for c in "0123456789":
            self.prefix_tree.setdefault(base_s + 4, {})[c] = base_s + 5

        for c in "0123456789":
            self.prefix_tree.setdefault(base_s + 5, {})[c] = base_s + 5

        self.state += 5
        return [base_s + 5]

    def build_int_state(self) -> list[int]:
        base_s = self.state
        f_nums = "-0123456789"
        for c in f_nums:
            if c == "-":
                self.prefix_tree.setdefault(base_s, {})[c] = base_s + 1
            elif c == "0":
                self.prefix_tree.setdefault(base_s, {})[c] = base_s + 2
            else:
                self.prefix_tree.setdefault(base_s, {})[c] = base_s + 3

        for c in "0123456789":
            if c == "0":
                self.prefix_tree.setdefault(base_s + 1, {})[c] = base_s + 2
            else:
                self.prefix_tree.setdefault(base_s + 1, {})[c] = base_s + 3

        for c in "0123456789":
            self.prefix_tree.setdefault(base_s + 3, {})[c] = base_s + 3

        self.state += 3
        return [base_s + 2, base_s + 3]

    def allowed_token(self, state: int) -> list[int]:
        allowed_tokens: list[int] = []

        if self.cache.get(state, None) is not None:
            return self.cache[state]

        if state not in self.prefix_tree:
            return []

        for key in self.prefix_tree[state]:
            if key is None:
                for token, token_id in self.all_vocab.items():
                    allowed_tokens.append(token_id)
                break

            if key not in self.vocab_pins:
                continue

            for token in self.vocab_pins[key]:
                tmp_state = state
                flag = True
                for c in token:
                    curr_transitions = self.prefix_tree.get(tmp_state, {})
                    if None in curr_transitions:
                        tmp_state = curr_transitions[None]
                    elif c in curr_transitions:
                        tmp_state = curr_transitions[c]
                        if tmp_state == -1:
                            flag = False
                            break
                    else:
                        flag = False
                        break
                if flag:
                    tid = self.vocab_pins[key][token]
                    if tid not in allowed_tokens:
                        allowed_tokens.append(tid)

        self.cache[state] = allowed_tokens
        return allowed_tokens

    def mask_logits(self, logits: Any, allowed_tokens: list[int]) -> list[float]:
        masked: list[float] = [float("-inf")] * len(logits)
        for token_id in allowed_tokens:
            if token_id < len(logits):
                masked[token_id] = logits[token_id]
        return masked

    def transition(self, token_str: str) -> None:
        for c in self.tr_token(token_str):
            if self.current_state == -1 or self.current_state not in self.prefix_tree:
                break

            edges = self.prefix_tree[self.current_state]

            if c in edges:
                self.current_state = edges[c]
            elif None in edges:
                self.current_state = edges[None]
            else:
                self.current_state = -1
                break