import numpy as np


def argmax(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def mask_logits_vectorized(
    logits: np.ndarray,
    current_prefix: str,
    allowed_choices: list[str],
    vocab_strings: np.ndarray,
) -> np.ndarray | None:
    """
    Vectorized validation that tests the entire vocabulary simultaneously.
    Returns None if current_prefix is already a complete valid choice.
    """
    if current_prefix in allowed_choices:
        return None

    matching_choices = [
        c for c in allowed_choices if c.startswith(current_prefix)
    ]
    if not matching_choices:
        return None

    # Identify non-empty token entries to prevent blank tokens from matching
    non_empty_tokens = vocab_strings != ""

    candidates = np.char.add(current_prefix, vocab_strings).astype(str)

    is_valid = np.zeros(len(logits), dtype=bool)

    for choice in matching_choices:
        # Check if candidate string can form the start of an allowed choice
        is_valid |= np.char.startswith(choice, candidates)

    # Ensure tokens with None/empty string in vocabulary are strictly excluded
    is_valid &= non_empty_tokens

    # If no valid token candidates exist, signal completion/failure safely
    if not np.any(is_valid):
        return None

    mask = np.full(len(logits), -np.inf, dtype=np.float32)
    mask[is_valid] = 0.0

    return logits + mask


def set_linear_layout(layout_str: str, idx_to_token: list) -> dict[int, int]:
    linear_index_cache: dict[int, int] = {}
    total_chars = len(layout_str)

    for state in range(total_chars):
        remaining = layout_str[state:]
        best_token_idx = -1
        max_len = 0

        for idx, token_str in enumerate(idx_to_token):
            if not token_str:
                continue
            length = len(token_str)
            if length > len(remaining) or length <= max_len:
                continue
            if remaining.startswith(token_str):
                max_len = length
                best_token_idx = idx

        linear_index_cache[state] = best_token_idx
    return linear_index_cache


def get_static_token_idx(cursor: int, linear_index_cache: dict[int, int]) -> int:
    return linear_index_cache.get(cursor, -1)


def pre_compile_args(chosen_func: str, functions):
    for fun in functions:
        if fun.name == chosen_func[:len(chosen_func)-2]:
            print(str(fun.parameter))
