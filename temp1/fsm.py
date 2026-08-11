from typing import Any


class FSM:
    """
    Finite-state machine for constrained function-call generation.
    """
    def __init__(
        self,
        fncs: list[Any],
        vocab_pins: dict[str, dict[str, int]],
        all_vocab: dict[str, int],
    ) -> None:
        """
        Initialize the finite-state machine.
        Args:
            fncs: Functions that the model is allowed to call.
            vocab_pins: Mapping between token prefixes and token IDs.
            all_vocab: Mapping of vocabulary tokens to token IDs.
        Returns:
            None.
        """
        self.state: int = 0
        self.prefix_tree: dict[int, dict[str | None, int]] = {}
        self.functions = fncs
        self.vocab_pins = vocab_pins
        self.all_vocab = all_vocab
        self.saved_state: int = 0
        self.current_state: int = 0
        self.cache: dict[int, list[int]] = {}

    def tr_token(self, string: str) -> str:
        """
        Converts spaces to 'Ġ' tokens for tokenizer compatibility.
        Returns:
            str.
        """
        return string.replace(" ", "Ġ")

    def build_state(self) -> None:
        """
        Build the prefix tree for valid function-call structures.
        """
        start = self.tr_token('{"name": "')
        params = self.tr_token('", "parameters": {')
        for c in start:
            self.prefix_tree.setdefault(self.state, {})[c] = self.state + 1
            self.state += 1

        self.saved_state = self.state

        for fnc in self.functions:
            curr_fnc_state = self.saved_state

            for c in self.tr_token(fnc.name):
                if c in self.prefix_tree.get(curr_fnc_state, {}):
                    curr_fnc_state = self.prefix_tree[curr_fnc_state][c]
                else:
                    next_state = self.state + 1
                    self.prefix_tree.setdefault(
                        curr_fnc_state, {}
                    )[c] = next_state
                    self.state = next_state
                    curr_fnc_state = self.state

            for c in params:
                if c in self.prefix_tree.get(curr_fnc_state, {}):
                    curr_fnc_state = self.prefix_tree[curr_fnc_state][c]
                else:
                    next_state = self.state + 1
                    self.prefix_tree.setdefault(
                        curr_fnc_state, {}
                    )[c] = next_state
                    self.state = next_state
                    curr_fnc_state = self.state

            num_params = len(fnc.parameters)

            for index, (key, param_obj) in enumerate(
                fnc.parameters.items(), 1
            ):
                prefix = f' "{key}": ' if index > 1 else f'"{key}": '

                for c in self.tr_token(prefix):
                    if c in self.prefix_tree.get(curr_fnc_state, {}):
                        curr_fnc_state = self.prefix_tree[curr_fnc_state][c]
                    else:
                        next_state = self.state + 1
                        self.prefix_tree.setdefault(
                            curr_fnc_state, {}
                        )[c] = next_state
                        self.state = next_state
                        curr_fnc_state = self.state

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
                    curr_fnc_state = self.state

    def build_type_state(self, param_obj: Any) -> list[int]:
        """
        Build the state transitions for a parameter type.
        Args:
            param_obj: Parameter whose type determines the state structure.
        Returns:
            Exit states generated for the parameter type.
        """
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
        """
        Build states that accept a JSON string value.
        Returns:
            The state that can be used after the string value.
        """
        entry_s = self.state

        self.prefix_tree.setdefault(entry_s, {})['"'] = entry_s + 1
        self.state += 1

        self.prefix_tree.setdefault(self.state, {})[None] = self.state
        self.prefix_tree[self.state]['"'] = self.state + 1
        self.state += 1

        return [self.state]

    def build_number_state(self) -> list[int]:
        """
        Build states that accept a JSON floating-point number.
        Returns:
            The final state for a valid number.
        """
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
        """
        Build states that accept a JSON integer value.
        Returns:
            The states representing valid integer endings.
        """
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

    def mask_logits(
        self,
        logits: Any,
        allowed_tokens: list[int]
    ) -> list[float]:
        masked: list[float] = [float("-inf")] * len(logits)

        for token_id in allowed_tokens:
            if token_id < len(logits):
                masked[token_id] = logits[token_id]

        return masked

    def transition(self, token_str: str) -> None:
        for c in self.tr_token(token_str):
            if (
                self.current_state == -1
                or self.current_state not in self.prefix_tree
            ):
                break

            edges = self.prefix_tree[self.current_state]

            if c in edges:
                self.current_state = edges[c]
            elif None in edges:
                self.current_state = edges[None]
            else:
                self.current_state = -1
                break
