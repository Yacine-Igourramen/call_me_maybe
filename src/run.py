import numpy as np


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
