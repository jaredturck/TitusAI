import os
import sqlite3
from contextlib import closing
from time import monotonic

import torch
import torch.distributed as dist
from torch.nn.functional import cross_entropy
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler
from tqdm import tqdm
from models import TitusAI

WINDOW_LENGTH = 512

class TokenWindows(Dataset):
    '''Read the pretraining corpus and sample contiguous windows.'''

    def __init__(self):
        '''Load the packed token IDs into one CPU tensor.'''
        packed = bytearray()
        with closing(sqlite3.connect("file:datasets.sqlite?mode=ro", uri=True)) as database:
            rows = database.execute("SELECT token_ids FROM passages NOT INDEXED WHERE stage = 'pretrain' ORDER BY id")
            for (token_ids,) in rows:
                packed.extend(token_ids)

        self.tokens = torch.frombuffer(packed, dtype=torch.int16)
        self.offsets = torch.arange(WINDOW_LENGTH + 1)

    def __len__(self):
        '''Return the number of possible window starts.'''
        return self.tokens.numel() - WINDOW_LENGTH

    def __getitem__(self, starts):
        '''Gather 513 tokens from each starting position.'''
        return self.tokens[torch.as_tensor(starts)[..., None] + self.offsets]

def train():
    '''Train TitusAI across GPUs and save its BF16 weights.'''
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()

    dataset = TokenWindows()
    samples_per_gpu = (dataset.tokens.numel() - 1) // (WINDOW_LENGTH * dist.get_world_size())
    sampler = RandomSampler(dataset, replacement=True, num_samples=samples_per_gpu)
    batch_sampler = BatchSampler(sampler, batch_size=180, drop_last=False)
    batches = DataLoader(dataset, batch_size=None, sampler=batch_sampler, pin_memory=True)

    model = TitusAI().to(device=device, dtype=torch.bfloat16)
    model = DistributedDataParallel(model, device_ids=[device.index])
    optimizer = AdamW(model.parameters(), lr=3e-4, fused=True)

    progress = tqdm(batches, desc="Training", unit="batch", disable=rank != 0)
    last_report = monotonic()
    for windows in progress:
        windows = windows.to(device=device, dtype=torch.long, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        logits = model(windows[:, :-1])
        targets = windows[:, 1:].reshape(-1)
        loss = cross_entropy(logits.flatten(0, 1).float(), targets)

        loss.backward()
        optimizer.step()

        if rank == 0 and monotonic() - last_report >= 10:
            progress.set_postfix(loss=f"{loss.item():.4f}")
            last_report = monotonic()

    if rank == 0:
        torch.save(model.module.state_dict(), "titusai.pt")
    dist.destroy_process_group()

if __name__ == "__main__":
    train()
