import numpy as np


def argmax(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def mask_logits_vectorized(logits: np.ndarray, current_prefix: str, allowed_choices: list[str], vocab_strings: np.ndarray) -> np.ndarray:
    """
    Vectorized validation that tests the entire vocabulary simultaneously.
    Has near-zero Python loop overhead.
    """
    matching_choices = [c for c in allowed_choices if c.startswith(current_prefix)]

    candidates = np.char.add(current_prefix, vocab_strings)

    is_valid = np.zeros(len(logits), dtype=bool)

    for choice in matching_choices:
        is_valid |= np.char.startswith(choice, candidates)

    mask = np.full(len(logits), -np.inf, dtype=np.float32)
    mask[is_valid] = 0.0

    return logits + mask


def set_linear_layout(layout_str: str, idx_to_token: list) -> dict[int, int]:
    """
    Pre-computes the absolute best single token index for every 
    character cursor offset inside the static layout string.
    """
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
    """
    Instantly returns the precompiled token index for a given layout position.
    """
    return linear_index_cache.get(cursor, -1)
