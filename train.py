''' Train the language model on one CUDA GPU. '''

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from model import LanguageModel

DEVICE = 'cuda'
DATA_PATH = Path('weights/data.bin')
MODEL_PATH = Path('weights/model.pt')
CONTEXT_LENGTH = 256
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
EPOCHS = 1

tokens = torch.from_file(str(DATA_PATH), shared=False, size=DATA_PATH.stat().st_size // 2, dtype=torch.uint16)
sequences = tokens.unfold(0, CONTEXT_LENGTH + 1, CONTEXT_LENGTH)
loader = DataLoader(sequences, batch_size=BATCH_SIZE, shuffle=True)

model = LanguageModel().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss()

print(f'Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}')
print(f'Training samples: {len(sequences):,}')

for epoch in range(EPOCHS):
    for step, batch in enumerate(loader):
        batch = batch.to(device=DEVICE, dtype=torch.long)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs)
        loss = loss_function(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f'epoch {epoch + 1} step {step:,} loss {loss.item():.4f}')

torch.save(model.state_dict(), MODEL_PATH)
