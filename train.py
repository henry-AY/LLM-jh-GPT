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

# x = train_data[:block_size]
# y = train_data[1:block_size + 1]
# for t in range(block_size):
#     context = x[:t + 1]
#     target = y[t]
#     print(f"when input is {context} the traget: {target}")

torch.manual_seed(1)
batch_size = 4

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('-----')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, : t + 1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target: {target}")

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets = None):
        #idx and targets are both (B, T) tensors of integers

        logits = self.token_embedding_table(idx) # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim = -1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T + 1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
print()
print("Without training: ")
print(decode(m.generate(torch.zeros((1, 1), dtype = torch.long), max_new_tokens = 200)[0].tolist()))

optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

batch_size = 32
for steps in range(100000):

    # sample a batch of data
    xb, yb = get_batch('train')

    #eval the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

print()

temp_loss = loss.item()
print("After training: ")
print(decode(m.generate(torch.zeros((1, 1), dtype = torch.long), max_new_tokens = 200)[0].tolist()))
print()
print("Ending loss of", temp_loss)
print()