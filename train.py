''' Train the language model on two CUDA GPUs. '''

import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from model import LanguageModel

WEIGHTS_PATH = Path('weights')
PRETRAIN_DATA_PATH = WEIGHTS_PATH / 'data.bin'
POSTTRAIN_DATA_PATH = WEIGHTS_PATH / 'posttrain.bin'
POSTTRAIN_MASK_PATH = WEIGHTS_PATH / 'posttrain_mask.bin'
PRETRAIN_CONTEXT_LENGTH = 256
POSTTRAIN_CONTEXT_LENGTH = 1024
PRETRAIN_BATCH_SIZE = 58
POSTTRAIN_BATCH_SIZE = 12
LEARNING_RATE = 3e-4
WARMUP_STEPS = 100
EPOCHS = 3
STOP_LOSS = 2.0

class Trainer:
    ''' Train the language model across two CUDA GPUs. '''

    def __init__(self, stage):
        ''' Initialize the trainer for pretraining or post-training. '''
        assert stage in ('pretrain', 'posttrain')
        self.stage = stage
        self.context_length = PRETRAIN_CONTEXT_LENGTH if stage == 'pretrain' else POSTTRAIN_CONTEXT_LENGTH
        self.batch_size = PRETRAIN_BATCH_SIZE if stage == 'pretrain' else POSTTRAIN_BATCH_SIZE
        self.checkpoint_prefix = 'model' if stage == 'pretrain' else 'posttrain'

        self.setup_distributed()
        self.setup_data()
        self.setup_training()

        if self.rank == 0:
            print(f'Parameters: {sum(parameter.numel() for parameter in self.model.parameters()):,}')
            print(f'Training samples: {len(self.dataset):,}')

        self.last_print = time.monotonic()
        self.last_print_step = -1
        self.last_save = self.last_print
        self.recent_losses = []
        self.stop = torch.zeros(1, device=self.device, dtype=torch.uint8)

    def setup_distributed(self):
        ''' Initialize distributed GPU training. '''
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')

        self.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device('cuda', self.local_rank)
        self.rank = dist.get_rank()

    def setup_loader(self):
        ''' Build the distributed data loader. '''
        self.sampler = DistributedSampler(self.dataset, shuffle=True, drop_last=True)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.sampler)

    def setup_pretrain_epoch(self, epoch):
        ''' Shift pretraining window boundaries for each epoch. '''
        offset = random.Random(epoch).randrange(self.context_length)
        self.dataset = self.tokens[offset:].unfold(0, self.context_length + 1, self.context_length)
        self.setup_loader()

    def setup_data(self):
        ''' Load and distribute training samples. '''
        if self.stage == 'pretrain':
            self.tokens = torch.from_numpy(np.fromfile(PRETRAIN_DATA_PATH, dtype=np.uint16))
            self.setup_pretrain_epoch(0)
        else:
            self.tokens = torch.from_file(str(POSTTRAIN_DATA_PATH), shared=False, size=POSTTRAIN_DATA_PATH.stat().st_size // 2, dtype=torch.uint16)
            self.masks = torch.from_file(str(POSTTRAIN_MASK_PATH), shared=False, size=POSTTRAIN_MASK_PATH.stat().st_size, dtype=torch.uint8)
            self.sequences = self.tokens.view(-1, self.context_length + 1)
            self.masks = self.masks.view(-1, self.context_length + 1)
            self.dataset = TensorDataset(self.sequences, self.masks)
            self.setup_loader()

    def setup_training(self):
        ''' Build the model, optimizer, loss, and learning-rate schedule. '''
        model = LanguageModel()

        if self.stage == 'posttrain':
            checkpoints = sorted(WEIGHTS_PATH.glob('model_*.pt'), key=lambda path: path.stat().st_mtime)
        else:
            checkpoints = sorted(WEIGHTS_PATH.glob('*.pt'), key=lambda path: path.stat().st_mtime)

        if checkpoints:
            # Every run intentionally warm-starts model weights while creating a fresh optimizer and schedule.
            weights = torch.load(checkpoints[-1], map_location='cpu')
            del weights['output.weight']
            model.load_state_dict(weights, strict=False)

        model = model.to(self.device)
        self.model = DistributedDataParallel(model, device_ids=[self.local_rank], broadcast_buffers=False, gradient_as_bucket_view=True, static_graph=True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, fused=True)
        self.loss_function = nn.CrossEntropyLoss()

        total_steps = len(self.loader) * EPOCHS
        warmup = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_STEPS)
        decay = CosineAnnealingLR(self.optimizer, T_max=total_steps - WARMUP_STEPS, eta_min=LEARNING_RATE * 0.1)
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, decay], milestones=[WARMUP_STEPS])

    def prepare_batch(self, batch):
        ''' Move a batch to the GPU and prepare its training targets. '''
        if self.stage == 'posttrain':
            batch, mask = batch
            mask = mask.to(device=self.device, dtype=torch.bool)
        else:
            mask = None

        batch = batch.to(device=self.device, dtype=torch.long)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        if mask is not None:
            targets = targets.masked_fill(~mask[:, 1:], -100)

        return inputs, targets

    def save(self):
        ''' Save model weights while keeping three checkpoints. '''
        models = sorted(WEIGHTS_PATH.glob(f'{self.checkpoint_prefix}_*.pt'), key=lambda path: path.stat().st_mtime)

        if len(models) < 3:
            path = WEIGHTS_PATH / f'{self.checkpoint_prefix}_{len(models) + 1}.pt'
        else:
            path = models[0]

        torch.save(self.model.module.state_dict(), path)

    def train(self):
        ''' Train the language model. '''
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for epoch in range(EPOCHS):
                if self.stage == 'pretrain' and epoch:
                    self.setup_pretrain_epoch(epoch)

                self.sampler.set_epoch(epoch)

                if self.rank == 0:
                    self.last_print = time.monotonic()
                    self.last_print_step = -1

                for step, batch in enumerate(self.loader):
                    inputs, targets = self.prepare_batch(batch)

                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits = self.model(inputs)
                        loss = self.loss_function(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    if self.rank == 0:
                        now = time.monotonic()

                        if now - self.last_print >= 20:
                            loss_value = loss.item()
                            self.recent_losses.append(loss_value)
                            self.recent_losses = self.recent_losses[-5:]
                            average_loss = sum(self.recent_losses) / len(self.recent_losses)
                            eta = int((len(self.loader) - step - 1) * (now - self.last_print) / (step - self.last_print_step))
                            print(f'epoch {epoch + 1} step {step:,} loss {loss_value:.4f} ({average_loss:.2f}) lr {self.scheduler.get_last_lr()[0]:.2e} eta {eta}s')
                            self.last_print = now
                            self.last_print_step = step

                            if len(self.recent_losses) == 5 and average_loss < STOP_LOSS:
                                self.stop.fill_(1)

                            if now - self.last_save >= 600:
                                self.save()
                                self.last_save = now

                    if step % 100 == 0:
                        dist.broadcast(self.stop, src=0)

                        if self.stop.item():
                            break

        if self.rank == 0:
            self.save()

        dist.barrier()

if __name__ == '__main__':
    stages = ('pretrain', 'posttrain') if sys.argv[1] == 'all' else (sys.argv[1],)

    for stage in stages:
        trainer = Trainer(stage)

        try:
            trainer.train()
        except KeyboardInterrupt:
            if trainer.rank == 0:
                trainer.save()
            break

        del trainer

    dist.destroy_process_group()
