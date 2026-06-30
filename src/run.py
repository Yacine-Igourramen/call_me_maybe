from llm_sdk import Small_LLM_Model
import json
import numpy as np

model_object = Small_LLM_Model()

prompt = "What is the sum of 265 and 345?"

# Read the vocabulary file
path = model_object.get_path_to_vocab_file()
with open(path, "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Create a fast array mapping token index directly to its string representation
# This eliminates a costly list(vocab.keys()) conversion inside the logit loop
dummy_logits = model_object.get_logits_from_input_ids([1])
vocab_size = len(dummy_logits)

# 2. Build the idx_to_token map explicitly matching that exact size
idx_to_token = [None] * vocab_size
for token_str, idx in vocab.items():
    if idx < vocab_size:
        # Perform the necessary string replacements so the vocabulary 
        # matches the raw string format used in your exp_output template
        sanitized = token_str

        # Convert literal character escape strings into actual whitespace bytes
        sanitized = sanitized.replace("Ċ", "\n")

        # Handle common tokenizer special space fragments (like SentencePiece/Llama marks)
        sanitized = sanitized.replace("Ġ", " ")

        idx_to_token[idx] = sanitized

# Our exact expected structure template 
# Note: In a real agent workflow, you would dynamically fill these template slots,
# but for this structure constraint engine, we are forcing this exact sequence.
exp_output = f'''
{{
"prompt": "{prompt}",
"fn_name": "",
"args": ""
}}
'''


class SglStructuralEngine:
    def __init__(self, exp_output: str, idx_to_token: list):
        self.schema = exp_output
        self.idx_to_token = idx_to_token
        self.vocab_size = len(idx_to_token)

        # Total number of states is the length of our target schema string
        self.total_states = len(exp_output)

        # Pre-compiled transition map: [current_state_idx] -> array of valid token IDs
        self.state_to_valid_tokens = {}

        print("Compiling vocabulary transition map... (This runs ONCE)")
        self._compile_transition_map()

    def _compile_transition_map(self):
        """
        Pre-calculates which tokens are valid at every single structural character offset.
        """
        for state in range(self.total_states):
            remaining_schema = self.schema[state:]
            valid_token_ids = []

            for idx, token_str in enumerate(self.idx_to_token):
                if token_str is None:
                    continue

                # Check if this token matches the expected structure sequence at this position
                if remaining_schema.startswith(token_str):
                    valid_token_ids.append(idx)
                # Handle cases where the remaining schema is shorter than the token itself
                elif token_str.startswith(remaining_schema) and state + len(remaining_schema) == self.total_states:
                    valid_token_ids.append(idx)

            self.state_to_valid_tokens[state] = np.array(valid_token_ids, dtype=np.int32)

    def mask_logits(self, logits: np.ndarray, current_state: int) -> np.ndarray:
        """
        Vectorized masking on the logits array using the pre-compiled transition index.
        """
        if current_state >= self.total_states:
            return logits  # Generation complete

        # Get the allowed token IDs for our current character cursor position
        allowed_indices = self.state_to_valid_tokens.get(current_state, np.array([], dtype=np.int32))

        # Create an all-infinite-negative mask
        mask = np.full(self.vocab_size, -np.inf, dtype=np.float32)

        if allowed_indices.size > 0:
            # Unmask only the mathematically valid token paths
            mask[allowed_indices] = 0.0

        # Add the mask to logits (valid indices get +0, invalid get -inf)
        return logits + mask


def argmax(logits: np.ndarray) -> int:
    """
    Finds the index of the largest element, safely handling -inf values.
    """
    return int(np.argmax(logits))


# 1. Prepare initial model context
feed = model_object.encode(prompt)[0].tolist()
output_str: str = ""

print("--- Starting Structured JSON Generation Loop ---")

engine = SglStructuralEngine(exp_output, idx_to_token)

# 2. Main Autoregressive Token Generation Loop
i = 0
while i < engine.total_states:
    # Get the raw model logits array
    logits = np.array(model_object.get_logits_from_input_ids(feed), dtype=np.float32)
    # Apply structural masking based on current text length pointer
    logits = engine.mask_logits(logits, i)

    # Choose the highest scoring allowed token
    next_token_idx = argmax(logits)
    chosen_token_str = idx_to_token[next_token_idx]

    # Update generation structures safely using the active index
    feed.append(next_token_idx)
    output_str += chosen_token_str

    print(f"Generated Token: {repr(chosen_token_str)} -> Current Result: {repr(output_str)}")

    # CRITICAL FIX: Advance the state index by the actual length of the token string
    i += len(chosen_token_str)

print("\n--- Final Forced Structured Output ---")
print(output_str)
