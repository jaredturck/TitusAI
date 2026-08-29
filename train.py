''' Train the language model on two CUDA GPUs. '''

import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler

from model import LanguageModel

DATA_PATH = Path('weights/data.bin')
MODEL_PATH = Path('weights/model.pt')
CONTEXT_LENGTH = 256
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WARMUP_STEPS = 100
EPOCHS = 1

def save(model):
    ''' Save model weights while keeping three checkpoints. '''
    models = sorted(MODEL_PATH.parent.glob('model_*.pt'), key=lambda path: path.stat().st_mtime)

    if len(models) < 3:
        path = MODEL_PATH.parent / f'model_{len(models) + 1}.pt'
    else:
        path = models[0]

    torch.save(model.module.state_dict(), path)

dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)
rank = dist.get_rank()

tokens = torch.from_file(str(DATA_PATH), shared=False, size=DATA_PATH.stat().st_size // 2, dtype=torch.uint16)
sequences = tokens.unfold(0, CONTEXT_LENGTH + 1, CONTEXT_LENGTH)
sampler = DistributedSampler(sequences, shuffle=True, drop_last=True)
loader = DataLoader(sequences, batch_size=BATCH_SIZE, sampler=sampler)

model = DistributedDataParallel(LanguageModel().to(device), device_ids=[local_rank])
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.CrossEntropyLoss()

total_steps = len(loader) * EPOCHS
warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_STEPS)
decay = CosineAnnealingLR(optimizer, T_max=total_steps - WARMUP_STEPS, eta_min=LEARNING_RATE * 0.1)
scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[WARMUP_STEPS])

if rank == 0:
    print(f'Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}')
    print(f'Training samples: {len(sequences):,}')

last_print = time.monotonic()
last_save = last_print

for epoch in range(EPOCHS):
    sampler.set_epoch(epoch)

    for step, batch in enumerate(loader):
        batch = batch.to(device=device, dtype=torch.long)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits = model(inputs)
        loss = loss_function(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        now = time.monotonic()

        if rank == 0 and now - last_print >= 20:
            print(f'epoch {epoch + 1} step {step:,} loss {loss.item():.4f} lr {scheduler.get_last_lr()[0]:.2e}')
            last_print = now

        if rank == 0 and now - last_save >= 600:
            save(model)
            last_save = now

if rank == 0:
    save(model)

dist.destroy_process_group()
