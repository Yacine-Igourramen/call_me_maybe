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
        self.linear_cache: dict[int, np.ndarray] = {}

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
        self.linear_cache.clear()
        total_chars = len(layout_str)

        for state in range(total_chars + 1):
            remaining = layout_str[state:]
            valid_ids = []

            if remaining:
                for idx, token_str in enumerate(self.idx_to_token):
                    if not token_str:
                        continue
                    if len(token_str) > len(remaining):
                        continue
                    if not remaining.startswith(token_str):
                        continue
                    valid_ids.append(idx)

            mask = self._base_mask.copy()
            if valid_ids:
                mask[valid_ids] = 0.0
            self.linear_cache[state] = mask

    def mask_logits_linear(
        self, logits: np.ndarray, cursor: int
    ) -> np.ndarray:
        # Defaults to the baseline -inf mask if the cursor falls out of bounds
        mask = self.linear_cache.get(cursor, self._base_mask)
        return logits + mask

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
