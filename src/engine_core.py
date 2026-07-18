import numpy as np


class TrieNode:
    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_end_of_word: bool = False


class Engine:
    def __init__(
        self, allowed_choices: list[str], idx_to_token: list[str | None]
    ) -> None:
        self.idx_to_token: list[str | None] = idx_to_token
        self.vocab_size = len(idx_to_token)

        # Pre-allocate a reusable base mask full of -inf
        self._base_mask = np.full(self.vocab_size, -np.inf, dtype=np.float32)

        self.trie_root = TrieNode()
        for choice in allowed_choices:
            self._insert_trie(choice)

        self.trie_cache: dict[TrieNode, np.ndarray] = {}
        self._precompile_trie(self.trie_root)
        
        # Maps layout cursor position directly to the exact single token index
        self.linear_index_cache: dict[int, int] = {}

    def _insert_trie(self, word: str) -> None:
        node = self.trie_root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def _precompile_trie(self, node: TrieNode) -> None:
        valid_ids = []
        for idx, token_str in enumerate(self.idx_to_token):
            if not token_str:
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
        mask = self._base_mask.copy()
        if valid_ids:
            mask[valid_ids] = 0.0
        self.trie_cache[node] = mask

        for child in node.children.values():
            self._precompile_trie(child)

    def set_linear_layout(self, layout_str: str) -> None:
        """
        Pre-computes the absolute best single token index for every 
        character cursor offset inside the static layout string.
        """
        self.linear_index_cache.clear()
        total_chars = len(layout_str)

        for state in range(total_chars):
            remaining = layout_str[state:]
            best_token_idx = -1
            max_len = 0

            for idx, token_str in enumerate(self.idx_to_token):
                if not token_str:
                    continue
                length = len(token_str)
                # Keep the longest matching token to prevent tokenizer fragmentation
                if length > len(remaining) or length <= max_len:
                    continue
                if remaining.startswith(token_str):
                    max_len = length
                    best_token_idx = idx

            self.linear_index_cache[state] = best_token_idx

    def get_static_token_idx(self, cursor: int) -> int:
        """
        Instantly returns the precompiled token index for a given layout position.
        """
        return self.linear_index_cache.get(cursor, -1)

    def mask_logits_trie(
        self, logits: np.ndarray, node: TrieNode
    ) -> np.ndarray:
        mask = self.trie_cache.get(node, self._base_mask)
        return logits + mask

    def advance_trie(self, node: TrieNode, token_str: str) -> TrieNode | None:
        curr = node
        for char in token_str:
            next_node = curr.children.get(char)
            if next_node is None:
                return None
            curr = next_node
        return curr


def argmax(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


vocab_strings = np.array([t if t else "" for t in idx_to_token], dtype=object)


def mask_logits_vectorized(logits: np.ndarray, current_prefix: str, allowed_choices: list[str], vocab_strings: np.ndarray) -> np.ndarray:
    """
    Vectorized validation that tests the entire vocabulary simultaneously.
    Has near-zero Python loop overhead.
    """
    # 1. Pre-filter choices matching the current prefix to minimize string matching work
    matching_choices = [c for c in allowed_choices if c.startswith(current_prefix)]
    
    # 2. Re-create the candidates array by prepending the current prefix to all tokens at once
    candidates = np.char.add(current_prefix, vocab_strings)
    
    # 3. Initialize a boolean array tracking valid slots (Default: False)
    is_valid = np.zeros(len(logits), dtype=bool)
    
    # 4. Use NumPy vector operations to check matches across all tokens instantly
    for choice in matching_choices:
        # If a candidate can form the start of this choice string, mark it true
        is_valid |= np.char.startswith(choice, candidates)
        
    # 5. Build and apply the mask vector instantly
    mask = np.full(len(logits), -np.inf, dtype=np.float32)
    mask[is_valid] = 0.0
    
    return logits + mask