import argparse, json, os, sqlite3, torch
from contextlib import closing
from time import monotonic

import torch.distributed as dist
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import cross_entropy
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler, RandomSampler
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

@sdpa_kernel(SDPBackend.FLASH_ATTENTION)
def train():
    '''Pretrain TitusAI and then fine-tune it on assistant replies.'''
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
    if os.path.isfile("model.pt"):
        model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model = DistributedDataParallel(model, device_ids=[device.index])
    optimizer = AdamW(model.parameters(), lr=3e-4, fused=True)

    progress = tqdm(batches, desc="Pretraining", unit="batch", disable=rank != 0)
    last_report = monotonic()
    for windows in progress:
        windows = windows.to(device=device, dtype=torch.long, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        logits = model(windows[:, :-1])
        targets = windows[:, 1:].reshape(-1)
        loss = cross_entropy(logits.flatten(0, 1).float(), targets)

        loss.backward()
        optimizer.step()

        if rank == 0 and monotonic() - last_report >= 5:
            progress.set_postfix(loss=f"{loss.item():.4f}")
            last_report = monotonic()

    if rank == 0:
        torch.save(model.module.state_dict(), "model.pt")

    del dataset, sampler, batch_sampler, batches, progress
    dataset = ChatDataset()
    sampler = DistributedSampler(dataset)
    batches = DataLoader(dataset, batch_size=32, sampler=sampler, pin_memory=True, collate_fn=pad_chat_batch)
    optimizer = AdamW(model.parameters(), lr=1e-4, fused=True)

    progress = tqdm(batches, desc="Fine-tuning", unit="batch", disable=rank != 0)
    last_report = monotonic()
    for inputs, targets in progress:
        inputs = inputs.to(device=device, dtype=torch.long, non_blocking=True)
        targets = targets.to(device=device, dtype=torch.long, non_blocking=True)
        optimizer.zero_grad()

        logits = model(inputs)
        loss = cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
        loss.backward()
        optimizer.step()

        if rank == 0 and monotonic() - last_report >= 5:
            progress.set_postfix(loss=f"{loss.item():.4f}")
            last_report = monotonic()

    if rank == 0:
        torch.save(model.module.state_dict(), "titusai-finetuned.pt")

    dist.destroy_process_group()

class ChatDataset(Dataset):
    '''Load tokenized conversations and their assistant reply targets.'''

    def __init__(self):
        '''Read post-training conversations from SQLite.'''
        self.examples = []
        with closing(sqlite3.connect("file:datasets.sqlite?mode=ro", uri=True)) as database:
            rows = database.execute("SELECT token_ids, source_metadata FROM passages WHERE stage = 'posttrain' ORDER BY id")
            for token_ids, metadata in rows:
                tokens = torch.frombuffer(bytearray(token_ids), dtype=torch.int16)
                targets = torch.full_like(tokens, -100)
                for start, end in json.loads(metadata)["assistant_spans"]:
                    targets[start:end] = tokens[start:end]
                self.examples.append((tokens[:-1], targets[1:]))

    def __len__(self):
        '''Return the number of conversations.'''
        return len(self.examples)

    def __getitem__(self, index):
        '''Return one conversation and its next-token targets.'''
        return self.examples[index]

def pad_chat_batch(examples):
    '''Right-pad the conversations and ignore padded loss positions.'''
    inputs, targets = zip(*examples)
    return pad_sequence(inputs, batch_first=True, padding_value=3), pad_sequence(targets, batch_first=True, padding_value=-100)

if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    train()
