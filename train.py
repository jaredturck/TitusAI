''' Train the language model on one CUDA GPU. '''

import torch
from torch import nn
from torch.utils.data import DataLoader

from model import CONTEXT_LENGTH, VOCAB_SIZE, LanguageModel

DEVICE = 'cuda'
DATA_PATH = 'weights/data.pt'
MODEL_PATH = 'weights/model.pt'
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
EPOCHS = 1

tokens = torch.load(DATA_PATH)
sequences = tokens.unfold(0, CONTEXT_LENGTH + 1, CONTEXT_LENGTH)
loader = DataLoader(sequences, batch_size=BATCH_SIZE, shuffle=True)

model = LanguageModel().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss()

print(f'Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}')

for epoch in range(EPOCHS):
    for step, batch in enumerate(loader):
        batch = batch.long().to(DEVICE)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs)
        loss = loss_function(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f'epoch {epoch + 1} step {step:,} loss {loss.item():.4f}')

torch.save(model.state_dict(), MODEL_PATH)
