"""
Character-level bigram language model that generates new first names.

Step 1 of building a small LLM from scratch:
tokenizer -> bigram counts -> probability matrix -> sampling.
"""

from pathlib import Path

import torch

DATA_PATH = Path(__file__).parent / "names.txt"
BOUNDARY_TOKEN = "."  # marks both the start and the end of a name
NUM_SAMPLES = 10


# --- Load the dataset -------------------------------------------------------
with open(DATA_PATH, "r", encoding="utf-8") as file:
    raw_text = file.read()

names = [line.strip() for line in raw_text.splitlines() if line.strip()]


# --- Tokenizer (1 character = 1 token) --------------------------------------
all_characters = [char for name in names for char in name]
all_characters.append(BOUNDARY_TOKEN)
vocab = sorted(set(all_characters))
vocab_size = len(vocab)

stoi = {char: index for index, char in enumerate(vocab)}   # string -> integer
itos = {index: char for index, char in enumerate(vocab)}   # integer -> string


def encode(text):
    """Convert a string into a list of token ids."""
    token_ids = []
    for char in text:
        token_ids.append(stoi[char])
    return token_ids


def decode(token_ids):
    """Convert a list of token ids back into a string."""
    chars = []
    for token_id in token_ids:
        chars.append(itos[token_id])
    return "".join(chars)


# --- Count bigrams ----------------------------------------------------------
padded_names = []
for name in names:
    padded_names.append(BOUNDARY_TOKEN + name + BOUNDARY_TOKEN)

bigrams = []
for name in padded_names:
    for current_char, next_char in zip(name, name[1:]):
        bigrams.append((current_char, next_char))

bigram_counts = {}
for bigram in bigrams:
    bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

top_bigrams = sorted(bigram_counts.items(), key=lambda item: item[1], reverse=True)


# --- Count matrix and probability matrix ------------------------------------
# counts[i, j] = number of times token i is followed by token j
counts = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
for (current_char, next_char), count in bigram_counts.items():
    counts[stoi[current_char], stoi[next_char]] += count

# probs[i] = probability distribution of the token that follows token i
probs = torch.zeros((vocab_size, vocab_size), dtype=torch.float32)
for row in range(vocab_size):
    probs[row] = counts[row] / counts[row].sum()


# --- Generate new names -----------------------------------------------------
if __name__ == "__main__":
    print("Top 10 bigrams:", top_bigrams[:10])
    print()

    boundary_id = stoi[BOUNDARY_TOKEN]
    for _ in range(NUM_SAMPLES):
        current_id = boundary_id
        generated_ids = []

        while True:
            next_id = torch.multinomial(probs[current_id], num_samples=1).item()
            if next_id == boundary_id:
                break
            generated_ids.append(next_id)
            current_id = next_id

        print(decode(generated_ids))
