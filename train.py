import torch
import torch.nn as nn
from torch.nn import functional as F


with open('english_input_developed_by_AI.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# temp
print("Number of characters in text file: ", len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)

# temp
print(''.join(chars))
print(vocab_size)

stoi = {ch : i for i, ch in enumerate(chars) }
itos = {i : ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, outut a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

#temp
print(encode("hi there"))
print(decode(encode("hi there")))

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[0:1000])

n = int(0.9*len(data)) # first 90% will train, rest validate
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size + 1]