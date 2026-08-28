import torch
from torch import nn
from torch.utils.data import DataLoader

from model import LanguageModel, context_length, vocab_size


device = 'cuda'
batch_size = 8
learning_rate = 3e-4
epochs = 1

tokens = torch.load('weights/data.pt')
sequences = tokens.unfold(0, context_length + 1, context_length)
loader = DataLoader(sequences, batch_size=batch_size, shuffle=True)

model = LanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

print(f'Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}')

for epoch in range(epochs):
    for step, batch in enumerate(loader):
        batch = batch.long().to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs)
        loss = loss_function(logits.reshape(-1, vocab_size), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f'epoch {epoch + 1} step {step:,} loss {loss.item():.4f}')

torch.save(model.state_dict(), 'weights/model.pt')
