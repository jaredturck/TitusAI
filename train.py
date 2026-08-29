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

class Trainer:
    ''' Train the language model across two CUDA GPUs. '''

    def __init__(self):
        ''' Initialize the trainer. '''
        self.setup_distributed()
        self.setup_data()
        self.setup_training()

        if self.rank == 0:
            print(f'Parameters: {sum(parameter.numel() for parameter in self.model.parameters()):,}')
            print(f'Training samples: {len(self.sequences):,}')

        self.last_print = time.monotonic()
        self.last_save = self.last_print

    def setup_distributed(self):
        ''' Initialize distributed GPU training. '''
        dist.init_process_group(backend='nccl')
        self.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device('cuda', self.local_rank)
        self.rank = dist.get_rank()

    def setup_data(self):
        ''' Load and distribute packed training samples. '''
        self.tokens = torch.from_file(str(DATA_PATH), shared=False, size=DATA_PATH.stat().st_size // 2, dtype=torch.uint16)
        self.sequences = self.tokens.unfold(0, CONTEXT_LENGTH + 1, CONTEXT_LENGTH)
        self.sampler = DistributedSampler(self.sequences, shuffle=True, drop_last=True)
        self.loader = DataLoader(self.sequences, batch_size=BATCH_SIZE, sampler=self.sampler)

    def setup_training(self):
        ''' Build the model, optimizer, loss, and learning-rate schedule. '''
        self.model = DistributedDataParallel(LanguageModel().to(self.device), device_ids=[self.local_rank])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        self.loss_function = nn.CrossEntropyLoss()

        total_steps = len(self.loader) * EPOCHS
        warmup = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_STEPS)
        decay = CosineAnnealingLR(self.optimizer, T_max=total_steps - WARMUP_STEPS, eta_min=LEARNING_RATE * 0.1)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, decay], milestones=[WARMUP_STEPS])

    def save(self):
        ''' Save model weights while keeping three checkpoints. '''
        models = sorted(MODEL_PATH.parent.glob('model_*.pt'), key=lambda path: path.stat().st_mtime)

        if len(models) < 3:
            path = MODEL_PATH.parent / f'model_{len(models) + 1}.pt'
        else:
            path = models[0]

        torch.save(self.model.module.state_dict(), path)

    def train(self):
        ''' Train the language model. '''
        for epoch in range(EPOCHS):
            self.sampler.set_epoch(epoch)

            for step, batch in enumerate(self.loader):
                batch = batch.to(device=self.device, dtype=torch.long)
                inputs = batch[:, :-1]
                targets = batch[:, 1:]

                logits = self.model(inputs)
                loss = self.loss_function(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                now = time.monotonic()

                if self.rank == 0 and now - self.last_print >= 20:
                    print(f'epoch {epoch + 1} step {step:,} loss {loss.item():.4f} lr {self.scheduler.get_last_lr()[0]:.2e}')
                    self.last_print = now

                if self.rank == 0 and now - self.last_save >= 600:
                    self.save()
                    self.last_save = now

        if self.rank == 0:
            self.save()

        dist.destroy_process_group()

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
