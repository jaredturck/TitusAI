import contextlib
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from model import TitusModel


def tiny_config():
    return {
        'vocab_size': 128,
        'd_model': 64,
        'num_layers': 2,
        'num_heads': 4,
        'num_kv_heads': 2,
        'head_dim': 16,
        'ffn_dim': 128,
        'max_seq_len': 32,
        'rms_norm_eps': 1e-5,
        'rope_theta': 10_000.0,
        'dropout': 0.0,
        'loss_chunk_size': 16,
    }


def main():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        backend = 'nccl'
    else:
        device = torch.device('cpu')
        backend = 'gloo'

    dist.init_process_group(backend)
    torch.manual_seed(100 + rank)

    model = TitusModel(tiny_config()).to(device)
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank] if device.type == 'cuda' else None,
        output_device=local_rank if device.type == 'cuda' else None,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)

    for micro_step in range(2):
        input_ids = torch.randint(0, 128, (2, 16), device=device)
        labels = torch.randint(0, 128, (2, 16), device=device)
        synchronization_context = contextlib.nullcontext()
        if micro_step == 0:
            synchronization_context = model.no_sync()

        with synchronization_context:
            loss = model(
                input_ids,
                labels=labels,
                return_logits=False,
            )['loss'] / 2
            loss.backward()

    optimizer.step()
    reduced_loss = torch.tensor(
        [float(loss.item())],
        device=device,
    )
    dist.all_reduce(reduced_loss)

    if rank == 0:
        print(f'[+] DDP smoke loss: {reduced_loss.item():.6f}')
        print('[+] DDP smoke test passed')

    dist.barrier()
    dist.destroy_process_group()
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
