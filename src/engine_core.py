import json
import re
from enum import Enum
import numpy as np


def argmax(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def mask_logits_vectorized(
    logits: np.ndarray,
    current_prefix: str,
    allowed_choices: list[str],
    vocab_strings: np.ndarray,
) -> np.ndarray | None:
    """Vectorized validation that tests the entire vocabulary simultaneously.

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


def get_static_token_idx(
    cursor: int, linear_index_cache: dict[int, int]
) -> int:
    return linear_index_cache.get(cursor, -1)


def pre_compile_args(
    chosen_func: str, functions: list, idx_to_token: list
) -> tuple[list[str], list[dict[int, int]], list[type]]:
    """Finds the target function schema, converts its Python parameter types

    back into JSON type strings, and pre-compiles the static layout cache.
    """
    target_function = None
    for fun in functions:
        if fun.name == chosen_func:
            target_function = fun
            break

    # Reverse mapping from Python type objects to JSON type strings
    reverse_type_map = {
        str: "string",
        float: "number",
        int: "integer",
        bool: "boolean",
    }

    # Handle functions with no parameters
    if not target_function or not target_function.parameter:
        layout_str = '",\n"parameters": {}\n}'
        return [layout_str], [set_linear_layout(layout_str, idx_to_token)], []

    param_list = []
    formatted_schema = {}
    for param_name, param_type in target_function.parameter.items():
        type_str = reverse_type_map.get(param_type, "string")
        formatted_schema[param_name] = type_str
        param_list.append(param_type)

    # Serialize to JSON fragment string
    params_json = json.dumps(formatted_schema, indent=2)
    layout_str = f'",\n"parameters": {params_json}\n}}'
    layout_str_chunks = re.split(
        r'string|"number"|"integer"|"boolean"', layout_str
    )

    linear_cache = [
        set_linear_layout(chunk, idx_to_token) for chunk in layout_str_chunks
    ]
    print(layout_str_chunks)
    return layout_str_chunks, linear_cache, param_list


class PrefixState(Enum):
    INVALID = 0
    PREFIX = 1
    COMPLETE = 2


# Pre-compiled RFC 8259 JSON grammar patterns for maximum loop performance
_INT_COMPLETE_RE = re.compile(r"-?(0|[1-9]\d*)")

_FLOAT_COMPLETE_RE = re.compile(
    r"-?(0|[1-9]\d*)(\.\d+([eE][+-]?\d+)?|[eE][+-]?\d+)"
)
_FLOAT_PREFIX_RE = re.compile(r"-?|-?(0|[1-9]\d*)(\.|(\.\d+)?[eE][+-]?)?")


def _is_valid_prefix(prefix: str, expected_type: type) -> PrefixState:
    # ---------------- STR ----------------
    if expected_type is str:
        return PrefixState.PREFIX

    # ---------------- BOOL ----------------
    if expected_type is bool:
        sl = prefix.lower()
        if sl in ("true", "false"):
            return PrefixState.COMPLETE
        if sl in ("", "t", "tr", "tru", "f", "fa", "fal", "fals"):
            return PrefixState.PREFIX
        return PrefixState.INVALID

    # ---------------- INT ----------------
    if expected_type is int:
        if _INT_COMPLETE_RE.fullmatch(prefix):
            return PrefixState.COMPLETE
        if prefix in ("", "-"):
            return PrefixState.PREFIX
        return PrefixState.INVALID

    # ---------------- FLOAT ----------------
    if expected_type is float:
        # 1. Complete float: must contain a decimal (.0, .123) OR exponent (e10, .0e10)
        if _FLOAT_COMPLETE_RE.fullmatch(prefix):
            return PrefixState.COMPLETE

        # 2. Incomplete float prefix:
        #    - empty "" or "-"
        #    - whole numbers waiting for a decimal: "16", "-0"
        #    - dangling decimal point: "16."
        #    - dangling scientific notation: "16e", "16.5e-"
        if _FLOAT_PREFIX_RE.fullmatch(prefix):
            return PrefixState.PREFIX

        return PrefixState.INVALID

    return PrefixState.INVALID


def mask_dynamic_args(
    logits: np.ndarray,
    expected_type: type,
    idx_to_token: list[str | None],
    current_value: str,
) -> np.ndarray | None:
    """Masks logits for a dynamic JSON value in pure token-by-token mode."""
    logits = np.array(logits, dtype=np.float32)
    masked = logits.astype(np.float32, copy=True)

    allowed = np.zeros(masked.shape, dtype=bool)
    state = _is_valid_prefix(current_value, expected_type)

    has_decimal = "." in current_value

    for idx, token in enumerate(idx_to_token):
        if not token:
            continue

        # ---------------- STRING ----------------
        if expected_type is str:
            if "\n" in token:
                continue
            allowed[idx] = True
            continue

        # ---------------- INT ----------------
        if expected_type is int:
            if "." in token:
                continue

            if token.startswith((",", "}", "\n")):
                if state == PrefixState.COMPLETE:
                    allowed[idx] = True
                continue

            candidate = current_value + token
            if _is_valid_prefix(candidate, expected_type) != PrefixState.INVALID:
                allowed[idx] = True
            continue

        # ---------------- FLOAT ----------------
        if expected_type is float:
            if "," in token:
                continue

            # If current value lacks a decimal/exponent, do not allow JSON separators yet.
            # Require the model to sample a '.' or decimal token first.
            if token.startswith((",", "}", "\n")):
                if state == PrefixState.COMPLETE and has_decimal:
                    allowed[idx] = True
                continue

            candidate = current_value + token
            if _is_valid_prefix(candidate, expected_type) != PrefixState.INVALID:
                allowed[idx] = True
            continue

        # ---------------- BOOL ----------------
        if token.startswith((",", "}", "\n")):
            if state == PrefixState.COMPLETE:
                allowed[idx] = True
            continue

        candidate = current_value + token
        if _is_valid_prefix(candidate, expected_type) != PrefixState.INVALID:
            allowed[idx] = True

    if not np.any(allowed):
        return None

    masked[~allowed] = -np.inf
    return masked


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def load_bpe_assets(vocab_path: str, merges_path: str):
    with open(vocab_path, "r", encoding="utf-8") as f:
        encoder = json.load(f)
    with open(merges_path, "r", encoding="utf-8") as f:
        bpe_data = f.read().split('\n')[1:-1]
    
    bpe_merges = [tuple(merge.split()) for merge in bpe_data if merge.strip()]
    bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
    byte_encoder = bytes_to_unicode()
    pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\sa-zA-Z0-9]+|\s+(?!\S)|\s+""")
    
    return encoder, bpe_ranks, byte_encoder, pat


def apply_bpe(token: str, bpe_ranks: dict):
    word = tuple(token)
    pairs = get_pairs(word)
    if not pairs:
        return token
    
    while True:
        bigram = min(pairs, key=lambda pair: bpe_ranks.get(pair, float('inf')))
        if bigram not in bpe_ranks:
            break
        
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except ValueError:
                new_word.extend(word[i:])
                break
            
            if word[i] == first and i < len(word)-1 and word[i+1] == second:
                new_word.append(first+second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
                
        word = tuple(new_word)
        if len(word) == 1:
            break
        pairs = get_pairs(word)
        
    return " ".join(word)


def encode_text(text: str, encoder: dict, bpe_ranks: dict, byte_encoder: dict, pat: re.Pattern) -> list[list[int]]:
    bpe_tokens = []
    for token in re.findall(pat, text):
        token = "".join(byte_encoder[b] for b in token.encode('utf-8'))
        for bpe_token in apply_bpe(token, bpe_ranks).split(" "):
            if bpe_token in encoder:
                bpe_tokens.append(encoder[bpe_token])
    return [bpe_tokens]
