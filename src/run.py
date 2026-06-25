from llm_sdk import Small_LLM_Model
import json
import numpy as np

model_object = Small_LLM_Model()

i = 0

prompt = "What is the sum of 265 and 345?"

layout = ['{"prompt": "', '",\n"fn_name": "', '",\n"args": "', '"}']
path = model_object.get_path_to_vocab_file()
with open(path, "r", encoding="utf-8") as f:
    vocab = json.load(f)



def mask_logits_by_state(logits: list, vocab: dict, current_state: str, token_emitted: str = "") -> tuple[np.ndarray, str]:
    """
    Masks logits based on an explicit state machine instead of a regex.
    States:
      'START' -> Expects exactly 'ID-' next.
      'DIGITS' -> Expects only digit tokens ('9', '5', etc.) next.
    """
    masked_logits = np.array(logits, dtype=np.float32)
    idx_to_token = {idx: token for token, idx in vocab.items()}
    # Update the tracking state based on the last token chosen
    if token_emitted == "ID-":
        current_state = "DIGITS"
    for idx in range(len(masked_logits)):
        token_str = idx_to_token.get(idx, "")
        if current_state == "START":
            # In the START state, only allow the structural prefix
            if token_str != "ID-":
                masked_logits[idx] = -np.inf
        elif current_state == "DIGITS":
            # In the DIGITS state, reject strings containing letters
            if not token_str.isdigit():
                masked_logits[idx] = -np.inf
    return masked_logits, current_state


def pre_decode(index, vocabulary=vocab):
    text = list(vocabulary.keys())

    return text[index]


def argmax(logits: list[float]):
    max = 0
    index = None
    for i, g in enumerate(logits):
        if g > max:
            max = g
            index = i
    return index


feed = model_object.encode(prompt)[0].tolist()
output: int = []
while (i < 50):
    logits = model_object.get_logits_from_input_ids(feed)
    next_token = argmax(logits)
    feed.append(int(list(vocab.values())[next_token]))
    output.append(pre_decode(next_token))
    i += 1

print(output)
