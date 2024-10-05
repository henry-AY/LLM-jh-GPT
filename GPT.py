import torch
import torch.nn as nn
from train import BigramLanguageModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('English/WarAndPeace.txt', 'r', encoding='utf-8') as f:
    text = f.read()

""" SET THE SEED UNIVERSALLY TO '1' """
torch.manual_seed(1)

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch : i for i, ch in enumerate(chars) }
itos = {i : ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, outut a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

model = BigramLanguageModel().to(device)
model.load_state_dict(torch.load("model/final_model_weights.pth"))
model.eval()

def generate_token(token, max_tokens):
    
    input_ids = torch.tensor([encode(token)], dtype=torch.long).to(device)

    generated_ids = model.generate(input_ids, max_new_tokens=max_tokens)

    return (decode(generated_ids[0].tolist()))

print(generate_token("War and Peace", max_tokens=500))