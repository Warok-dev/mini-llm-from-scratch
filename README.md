# First LLM — Name Generator

A tiny **character-level bigram language model** that invents new first names, built from scratch in Python and PyTorch.

This is the first step of a personal project: building a small LLM step by step (tokenizer → bigram model → neural network → attention → Transformer) to understand how large language models really work.

## How it works

1. **Tokenizer** — each character is a token (26 letters + `.`, which marks the start and end of a name → vocabulary of 27).
2. **Bigram counts** — every name is padded as `.emma.` and every pair of consecutive characters is counted (228,146 bigrams).
3. **Probability matrix** — a 27 × 27 matrix where row `i` is the probability distribution of the character that follows character `i`.
4. **Sampling** — start from `.`, sample the next character with `torch.multinomial`, repeat until `.` is sampled again.

## Example output

```
kahenn
wan
monn
eiriyay
jananaeynic
```

The names look plausible but odd: a bigram model only sees the **last** character, so it has no memory of what it already wrote. Fixing this (more context, then attention) is the next step.

## Run it

```bash
pip install torch
cd name_generator
python bigram.py
```

## Credits

- Dataset (`names.txt`) and the overall approach come from Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) project (MIT license).

## Roadmap

- [x] Character tokenizer
- [x] Bigram model (counting)
- [ ] Bigram model as a neural network (gradient descent)
- [ ] Longer context
- [ ] Self-attention
- [ ] Transformer
