import numpy as np


def argmax(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def mask_logits_vectorized(
    logits: np.ndarray,
    current_prefix: str,
    allowed_choices: list[str],
    idx_to_token: np.ndarray,
) -> np.ndarray | None:
    if current_prefix in allowed_choices:
        return None

    matching_choices = [
        c for c in allowed_choices if c.startswith(current_prefix)
    ]
    if not matching_choices:
        return None

    vocab_strings = np.array(idx_to_token, dtype=str)

    is_valid = np.zeros(len(logits), dtype=bool)
    prefix_len = len(current_prefix)

    for choice in matching_choices:
        remainder = choice[prefix_len:]
        is_valid |= np.char.startswith(remainder, vocab_strings)
    is_valid &= (vocab_strings != "")
    if not np.any(is_valid):
        return None
    return np.where(is_valid, logits, -np.inf)


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


def get_static_token_idx(
    cursor: int, linear_index_cache: dict[int, int]
) -> int:
    return linear_index_cache.get(cursor, -1)
