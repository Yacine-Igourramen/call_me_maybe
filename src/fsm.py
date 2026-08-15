from typing import Any, Optional, ClassVar
from pydantic import BaseModel, Field, ConfigDict
from llm_sdk import Small_LLM_Model
import numpy as np
import json


class FSM(BaseModel):
    """
    Finite-state machine for constrained JSON parameter generation.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
    )

    start_string: str
    allowed_choices: list[str]
    context: str

    state: int = 0
    current_state: int = 0

    prefix_tree: dict[int, dict[Optional[str], int]] = Field(
        default_factory=dict)
    cache: dict[int, list[int]] = Field(
        default_factory=dict)

    model_object: ClassVar[Small_LLM_Model] = Small_LLM_Model()
    all_vocab: dict[str, int] = Field(
        default_factory=dict)
    vocab_pins: dict[str, dict[str, int]] = Field(default_factory=dict)
    feed: list[int] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        self.feed = self.model_object.encode(self.context)[0].tolist()
        with open(self.model_object.get_path_to_vocab_file(), "r") as f:
            self.all_vocab: dict[str, int] = json.load(f)
        for tstr, tid in self.all_vocab.items():
            if not tstr:
                continue
            key = tstr[0]
            self.vocab_pins.setdefault(key, {})[tstr] = tid
        self.build_static_state(self.start_string)

    def tr_token(self, string: str) -> str:
        return string.replace(" ", "Ġ").replace("\n", "Ċ")

    def build_static_state(self, string: str) -> None:
        self.state = 0
        self.current_state = 0
        self.prefix_tree.clear()
        self.cache.clear()
        start = self.tr_token(string)
        for c in start:
            self.prefix_tree.setdefault(self.state, {})[c] = self.state + 1
            self.state += 1

    def build_func_state(self) -> None:
        self.state = 0
        self.current_state = 0
        self.prefix_tree.clear()
        self.cache.clear()
        for name in self.allowed_choices:
            t_state = self.state
            for c in name:
                self.prefix_tree.setdefault(t_state, {})[c] = t_state + 1
                t_state += 1

    def build_arg_state(self, parameters: dict) -> None:
        num_params = len(parameters)
        self.state = 0
        self.current_state = 0
        self.prefix_tree.clear()
        self.cache.clear()
        if num_params == 0:
            tmp = self.prefix_tree.setdefault(self.state, {})
            tmp["}"] = self.state + 1
            tmp2 = self.prefix_tree.setdefault(self.state + 1, {})
            tmp2["}"] = self.state + 2
            self.state += 2
            return

        for index, (key, param_obj) in enumerate(parameters.items(), 1):
            prefix = f' "{key}": ' if index > 1 else f'"{key}": '
            for c in self.tr_token(prefix):
                tmp = self.prefix_tree.setdefault(self.state, {})
                tmp[c] = self.state + 1
                self.state += 1

            exit_states = self.build_type_state(param_obj)
            is_last_param = index == num_params

            if is_last_param:
                brace_1 = self.state + 1
                for exit_s in exit_states:
                    tmp = self.prefix_tree.setdefault(exit_s, {})
                    tmp["}"] = brace_1

                brace_2 = brace_1 + 1
                tmpb = self.prefix_tree.setdefault(brace_1, {})
                tmpb["}"] = brace_2
                self.state = brace_2
            else:
                next_s = self.state + 1
                for exit_s in exit_states:
                    self.prefix_tree.setdefault(exit_s, {})[","] = next_s
                self.state = next_s

    def build_type_state(self, param_obj: Any) -> list[int]:
        if param_obj == "integer":
            return self.build_int_state()
        elif param_obj == "number":
            return self.build_number_state()
        elif param_obj == "boolean":
            return self.build_bool_state()
        else:
            return self.build_string_state()

    def build_string_state(self) -> list[int]:
        entry_s = self.state

        tmp = self.prefix_tree.setdefault(entry_s, {})
        tmp['"'] = entry_s + 1
        tmp2 = self.prefix_tree.setdefault(entry_s + 1, {})
        tmp2[None] = entry_s + 1
        tmp2['"'] = entry_s + 3
        tmp2['\\'] = entry_s + 2

        tmp3 = self.prefix_tree.setdefault(entry_s + 2, {})
        tmp3[None] = entry_s + 1

        self.state += 3

        return [self.state]

    def build_number_state(self) -> list[int]:
        base_s = self.state
        f_nums = "-0123456789"
        for c in f_nums:
            if c == "-":
                tmp = self.prefix_tree.setdefault(base_s, {})
                tmp[c] = base_s + 1
            elif c == "0":
                tmp = self.prefix_tree.setdefault(base_s, {})
                tmp[c] = base_s + 2
            else:
                tmp = self.prefix_tree.setdefault(base_s, {})
                tmp[c] = base_s + 3

        for c in "0123456789":
            tmp = self.prefix_tree.setdefault(base_s + 1, {})
            if c == "0":
                tmp[c] = base_s + 2
            else:
                tmp[c] = base_s + 3

        tmp = self.prefix_tree.setdefault(base_s + 2, {})
        tmp["."] = base_s + 4
        tmp2 = self.prefix_tree.setdefault(base_s + 3, {})
        tmp2["."] = base_s + 4

        for c in "0123456789":
            tmp = self.prefix_tree.setdefault(base_s + 3, {})
            tmp[c] = base_s + 3

        for c in "0123456789":
            tmp = self.prefix_tree.setdefault(base_s + 4, {})
            tmp[c] = base_s + 5

        for c in "0123456789":
            tmp = self.prefix_tree.setdefault(base_s + 5, {})
            tmp[c] = base_s + 5

        self.state += 5
        return [base_s + 5]

    def build_int_state(self) -> list[int]:
        base_s = self.state
        f_nums = "-0123456789"
        for c in f_nums:
            if c == "-":
                tmp = self.prefix_tree.setdefault(base_s, {})
                tmp[c] = base_s + 1
            elif c == "0":
                tmp = self.prefix_tree.setdefault(base_s, {})
                tmp[c] = base_s + 2
            else:
                tmp = self.prefix_tree.setdefault(base_s, {})
                tmp[c] = base_s + 3

        for c in "0123456789":
            tmp = self.prefix_tree.setdefault(base_s + 1, {})
            if c == "0":
                tmp[c] = base_s + 2
            else:
                tmp[c] = base_s + 3

        for c in "0123456789":
            tmp = self.prefix_tree.setdefault(base_s + 3, {})
            tmp[c] = base_s + 3

        self.state += 3
        return [base_s + 2, base_s + 3]

    def build_bool_state(self) -> list[int]:
        entry_s = self.state
        next_state = self.state + 1
        exits: list[int] = []

        for option in ("true", "false"):
            t_state = entry_s
            for c in option:
                self.prefix_tree.setdefault(t_state, {})[c] = next_state
                t_state = next_state
                next_state += 1
            exits.append(t_state)

        self.state = next_state
        return exits

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

    def mask_logits(
            self, allowed_tokens: list[int]) -> tuple[int, str]:
        logits = self.model_object.get_logits_from_input_ids(self.feed)
        masked: list[float] = [-np.inf] * len(logits)
        for token_id in allowed_tokens:
            if token_id < len(logits):
                masked[token_id] = logits[token_id]
        chosen: int = int(np.argmax(masked))
        self.feed.append(chosen)
        return chosen, self.model_object.decode([chosen])

    def transition(self, token_str: str) -> None:
        for c in self.tr_token(token_str):
            if self.current_state == -1:
                break
            if self.current_state not in self.prefix_tree:
                break

            edges = self.prefix_tree[self.current_state]

            if c in edges:
                self.current_state = edges[c]
            elif None in edges:
                self.current_state = edges[None]
            else:
                if c in ("Ġ", "Ċ"):
                    continue
                self.current_state = -1
                break
