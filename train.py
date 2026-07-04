import contextlib
import math
import os
import random
import socket
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from checkpoint import (
    capture_rng_state,
    find_latest_checkpoint,
    save_inference_snapshot,
    save_training_checkpoint,
    load_training_checkpoint,
)
from config import (
    CHECKPOINT_PATH,
    DISCORD_CONFIG,
    MODEL_CONFIG,
    SNAPSHOT_PATH,
    TRAIN_CONFIG,
)
from dataset import ShardShuffleSampler, TitusDataset
from model import TitusModel
from notifications import DiscordNotifier


def format_duration(seconds):
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)

    if days:
        return f'{days}d {hours:02d}h {minutes:02d}m'
    if hours:
        return f'{hours}h {minutes:02d}m {seconds:02d}s'
    return f'{minutes}m {seconds:02d}s'


def format_progress_bar(progress, width=18):
    progress = min(max(progress, 0.0), 100.0)
    filled = round(width * progress / 100)
    return '█' * filled + '░' * (width - filled)


def training_status_fields(run_name, global_step, tokens_seen, loss, smoothed_loss, validation_loss, learning_rate, tokens_per_second, elapsed_seconds, max_train_tokens, snapshot_name, peak_gpu_memory):
    progress = 100 * tokens_seen / max_train_tokens
    remaining_tokens = max(0, max_train_tokens - tokens_seen)
    eta = 'Calculating'
    if tokens_per_second > 0:
        eta = format_duration(remaining_tokens / tokens_per_second)

    return [
        ('Run', run_name),
        ('Optimizer step', f'{global_step:,}'),
        ('Current loss', 'Unavailable' if loss is None else f'{loss:.4f}'),
        ('Smoothed loss', 'Unavailable' if smoothed_loss is None else f'{smoothed_loss:.4f}'),
        ('Validation loss', 'Not run yet' if validation_loss is None else f'{validation_loss:.4f}'),
        ('Learning rate', f'{learning_rate:.3e}'),
        ('Tokens processed', f'{tokens_seen:,} / {max_train_tokens:,}', False),
        ('Throughput', f'{tokens_per_second:,} tokens/s'),
        ('Elapsed', format_duration(elapsed_seconds)),
        ('ETA', eta),
        ('Peak GPU memory', peak_gpu_memory),
        ('Latest snapshot', snapshot_name, False),
    ]


def training_status_description(tokens_seen, max_train_tokens):
    progress = 100 * tokens_seen / max_train_tokens
    return f'`{format_progress_bar(progress)}` **{progress:.2f}% complete**'


def format_peak_gpu_memory(device):
    if device.type != 'cuda':
        return 'CPU training'
    memory_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    return f'{memory_gb:.2f} GB on rank 0'


class WarmupCosineScheduler:
    def __init__(self, optimizer, base_lr, min_lr, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.step_number = 0
        self.set_learning_rate(self.learning_rate_for_step(0))

    def learning_rate_for_step(self, step):
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps

        progress = (step - self.warmup_steps) / (
            self.total_steps - self.warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + cosine * (self.base_lr - self.min_lr)

    def set_learning_rate(self, learning_rate):
        for group in self.optimizer.param_groups:
            group['lr'] = learning_rate

    def step(self):
        self.step_number += 1
        self.set_learning_rate(
            self.learning_rate_for_step(self.step_number)
        )

    def get_last_lr(self):
        return [self.optimizer.param_groups[0]['lr']]

    def state_dict(self):
        return {
            'step_number': self.step_number,
            'base_lr': self.base_lr,
            'min_lr': self.min_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
        }

    def load_state_dict(self, state):
        self.step_number = state['step_number']
        self.set_learning_rate(
            self.learning_rate_for_step(self.step_number)
        )


def setup_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        backend = 'nccl'
    else:
        device = torch.device('cpu')
        backend = 'gloo'

    if world_size > 1:
        dist.init_process_group(backend=backend)

    return rank, local_rank, world_size, device


def cleanup_distributed(world_size):
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed, rank):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_optimizer(model, device):
    decay_parameters = []
    no_decay_parameters = []

    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    parameter_groups = [
        {
            'params': decay_parameters,
            'weight_decay': TRAIN_CONFIG['weight_decay'],
        },
        {
            'params': no_decay_parameters,
            'weight_decay': 0.0,
        },
    ]

    return torch.optim.AdamW(
        parameter_groups,
        lr=TRAIN_CONFIG['learning_rate'],
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=device.type == 'cuda',
    )


def create_loader(dataset, rank, world_size, training):
    sampler = ShardShuffleSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=training,
        seed=TRAIN_CONFIG['seed'],
    )

    loader_arguments = {
        'dataset': dataset,
        'batch_size': TRAIN_CONFIG['micro_batch_size'],
        'sampler': sampler,
        'shuffle': False,
        'drop_last': training,
        'num_workers': TRAIN_CONFIG['num_workers'],
        'pin_memory': TRAIN_CONFIG['pin_memory'],
    }

    if TRAIN_CONFIG['num_workers'] > 0:
        loader_arguments['persistent_workers'] = TRAIN_CONFIG['persistent_workers']
        loader_arguments['prefetch_factor'] = TRAIN_CONFIG['prefetch_factor']

    return DataLoader(**loader_arguments), sampler


def reduce_sum(value, device, world_size):
    tensor = torch.tensor(value, device=device, dtype=torch.long)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def gather_rng_states(rank, world_size):
    local_state = capture_rng_state()
    if world_size == 1:
        return [local_state]

    gathered_states = [None] * world_size if rank == 0 else None
    dist.gather_object(local_state, gathered_states, dst=0)
    return gathered_states


def validate(model, validation_loader, device, world_size):
    model.eval()
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.float64)

    with torch.inference_mode():
        for batch_index, batch in enumerate(validation_loader):
            if batch_index >= TRAIN_CONFIG['validation_batches']:
                break

            input_ids = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            segment_ids = None
            if TRAIN_CONFIG['isolate_packed_documents']:
                segment_ids = batch['segment_ids'].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == 'cuda',
            ):
                output = model(
                    input_ids,
                    labels=labels,
                    segment_ids=segment_ids,
                    return_logits=False,
                )

            total_loss += output['loss_sum'].double()
            total_tokens += output['loss_tokens'].double()

    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    model.train()
    return float((total_loss / total_tokens.clamp_min(1)).item())


def save_full_checkpoint(model, optimizer, scheduler, model_config, checkpoint_path, rank, world_size, global_step, tokens_seen, best_validation_loss, epoch, samples_seen_in_epoch):
    rng_states = gather_rng_states(rank, world_size)

    if rank == 0:
        destination = save_training_checkpoint(
            model,
            optimizer,
            scheduler,
            model_config,
            TRAIN_CONFIG,
            checkpoint_path,
            TRAIN_CONFIG['checkpoint_keep'],
            global_step,
            tokens_seen,
            best_validation_loss,
            epoch,
            samples_seen_in_epoch,
            rng_states,
        )
        print(f'[+] Saved training checkpoint: {destination.name}')

    if world_size > 1:
        dist.barrier()


def load_initial_weights(model, weights_path):
    if weights_path is None:
        return

    weights = torch.load(
        weights_path,
        map_location='cpu',
        weights_only=False,
    )
    model.load_state_dict(weights['model'])
    print(f'[+] Initialized model weights from {weights_path}')


def main():
    rank, local_rank, world_size, device = setup_distributed()
    notifier = DiscordNotifier(DISCORD_CONFIG) if rank == 0 else None
    seed_everything(TRAIN_CONFIG['seed'], rank)

    torch.set_float32_matmul_precision('high')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    run_name = TRAIN_CONFIG['run_name']
    train_data_path = TRAIN_CONFIG['train_data_path']
    validation_data_path = TRAIN_CONFIG['validation_data_path']
    snapshot_path = SNAPSHOT_PATH / run_name
    checkpoint_path = CHECKPOINT_PATH / run_name

    train_dataset = TitusDataset(train_data_path)
    validation_dataset = TitusDataset(validation_data_path)
    assert len(train_dataset) > 0
    assert len(validation_dataset) > 0
    assert train_dataset.sequence_length == MODEL_CONFIG['max_seq_len']
    assert validation_dataset.sequence_length == MODEL_CONFIG['max_seq_len']

    model_config = dict(MODEL_CONFIG)
    model_config['vocab_size'] = train_dataset.manifest['tokenizer']['vocab_size']

    model = TitusModel(model_config).to(device)
    model.enable_gradient_checkpointing(
        TRAIN_CONFIG['gradient_checkpointing']
    )
    load_initial_weights(model, TRAIN_CONFIG['initial_weights'])
    optimizer = create_optimizer(model, device)

    tokens_per_update = (
        TRAIN_CONFIG['micro_batch_size']
        * model_config['max_seq_len']
        * TRAIN_CONFIG['gradient_accumulation_steps']
        * world_size
    )
    total_steps = math.ceil(
        TRAIN_CONFIG['max_train_tokens'] / tokens_per_update
    )
    warmup_steps = int(total_steps * TRAIN_CONFIG['warmup_ratio'])
    scheduler = WarmupCosineScheduler(
        optimizer,
        TRAIN_CONFIG['learning_rate'],
        TRAIN_CONFIG['min_learning_rate'],
        warmup_steps,
        total_steps,
    )

    global_step = 0
    tokens_seen = 0
    best_validation_loss = float('inf')
    epoch = 0
    samples_seen_in_epoch = 0

    latest_checkpoint = None
    if TRAIN_CONFIG['resume_training']:
        latest_checkpoint = find_latest_checkpoint(checkpoint_path)

    if latest_checkpoint is not None:
        checkpoint_data = load_training_checkpoint(
            latest_checkpoint,
            model,
            optimizer,
            scheduler,
            rank,
        )
        global_step = checkpoint_data['global_step']
        tokens_seen = checkpoint_data['tokens_seen']
        best_validation_loss = checkpoint_data['best_validation_loss']
        epoch = checkpoint_data['epoch']
        samples_seen_in_epoch = checkpoint_data.get('samples_seen_in_epoch', 0)
        if rank == 0:
            print(f'[+] Resumed {latest_checkpoint.name}')

    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == 'cuda' else None,
            output_device=local_rank if device.type == 'cuda' else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )

    train_loader, train_sampler = create_loader(
        train_dataset,
        rank,
        world_size,
        training=True,
    )
    validation_loader, validation_sampler = create_loader(
        validation_dataset,
        rank,
        world_size,
        training=False,
    )

    if rank == 0:
        parameter_count = model.module.parameter_count() if hasattr(model, 'module') else model.parameter_count()
        gpu_names = ', '.join(
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ) or 'CPU'
        resume_status = latest_checkpoint.name if latest_checkpoint is not None else 'New run'

        print(f'[+] Run: {run_name}')
        print(f'[+] Parameters: {parameter_count:,}')
        print(f'[+] Training sequences: {len(train_dataset):,}')
        print(f'[+] Validation sequences: {len(validation_dataset):,}')
        print(f'[+] World size: {world_size}')
        print(f'[+] Planned optimizer steps: {total_steps:,}')

        notifier.send(
            '🚀 TitusAI training started',
            [
                ('Run', run_name),
                ('Host', socket.gethostname()),
                ('Mode', 'Resumed' if latest_checkpoint is not None else 'Fresh run'),
                ('Model', f'{parameter_count:,} parameters'),
                ('Context length', f'{model_config["max_seq_len"]:,} tokens'),
                ('Devices', gpu_names, False),
                ('DDP processes', str(world_size)),
                ('Effective batch', f'{tokens_per_update:,} tokens/update'),
                ('Planned steps', f'{total_steps:,}'),
                ('Training sequences', f'{len(train_dataset):,}'),
                ('Target tokens', f'{TRAIN_CONFIG["max_train_tokens"]:,}'),
                ('Checkpoint', resume_status, False),
            ],
            description='Training is online and Discord monitoring is active.',
            color='blue',
        )

    optimizer.zero_grad(set_to_none=True)
    training_start = time.monotonic()
    snapshot_timer = training_start
    status_timer = training_start
    log_timer = training_start
    log_tokens = tokens_seen
    accumulated_local_tokens = 0
    accumulated_local_samples = 0
    accumulation_step = 0
    last_validation_loss = None
    latest_loss = None
    smoothed_loss = None
    latest_tokens_per_second = 0
    latest_snapshot_name = 'Not saved yet'

    try:
        while tokens_seen < TRAIN_CONFIG['max_train_tokens']:
            token_limit_reached = False
            train_sampler.set_epoch(epoch)
            train_sampler.set_start_index(samples_seen_in_epoch)
            validation_sampler.set_epoch(epoch)
            validation_sampler.set_start_index(0)

            for batch in train_loader:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                segment_ids = None
                if TRAIN_CONFIG['isolate_packed_documents']:
                    segment_ids = batch['segment_ids'].to(device, non_blocking=True)

                accumulation_step += 1
                should_update = (
                    accumulation_step
                    == TRAIN_CONFIG['gradient_accumulation_steps']
                )

                synchronization_context = contextlib.nullcontext()
                if world_size > 1 and not should_update:
                    synchronization_context = model.no_sync()

                with synchronization_context:
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == 'cuda',
                    ):
                        output = model(
                            input_ids,
                            labels=labels,
                            segment_ids=segment_ids,
                            return_logits=False,
                        )
                        loss = output['loss'] / TRAIN_CONFIG['gradient_accumulation_steps']

                    loss.backward()

                accumulated_local_tokens += int(output['loss_tokens'].item())
                accumulated_local_samples += input_ids.size(0)

                if not should_update:
                    continue

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    TRAIN_CONFIG['gradient_clip'],
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_tokens = reduce_sum(
                    accumulated_local_tokens,
                    device,
                    world_size,
                )
                tokens_seen += global_tokens
                global_step += 1
                samples_seen_in_epoch += accumulated_local_samples
                accumulated_local_tokens = 0
                accumulated_local_samples = 0
                accumulation_step = 0

                now = time.monotonic()
                if rank == 0:
                    latest_loss = float(output['loss'].item())

                if rank == 0 and now - log_timer >= TRAIN_CONFIG['log_interval_seconds']:
                    if smoothed_loss is None:
                        smoothed_loss = latest_loss
                    else:
                        smoothed_loss = 0.95 * smoothed_loss + 0.05 * latest_loss

                    token_delta = tokens_seen - log_tokens
                    latest_tokens_per_second = int(token_delta / (now - log_timer))
                    progress = 100 * tokens_seen / TRAIN_CONFIG['max_train_tokens']
                    learning_rate = scheduler.get_last_lr()[0]
                    print(
                        f'[+] step={global_step:,} tokens={tokens_seen:,} '
                        f'loss={latest_loss:.4f} lr={learning_rate:.3e} '
                        f'tps={latest_tokens_per_second:,} progress={progress:.2f}%'
                    )
                    log_timer = now
                    log_tokens = tokens_seen

                if now - snapshot_timer >= TRAIN_CONFIG['snapshot_interval_seconds']:
                    if world_size > 1:
                        dist.barrier()
                    if rank == 0:
                        destination = save_inference_snapshot(
                            model,
                            model_config,
                            train_dataset.manifest['tokenizer'],
                            snapshot_path,
                            TRAIN_CONFIG['snapshot_keep'],
                            global_step,
                            tokens_seen,
                            last_validation_loss,
                        )
                        latest_snapshot_name = destination.name
                        print(f'[+] Saved inference snapshot: {destination.name}')
                    if world_size > 1:
                        dist.barrier()
                    snapshot_timer = time.monotonic()

                if rank == 0 and now - status_timer >= DISCORD_CONFIG['status_interval_seconds']:
                    notifier.send(
                        '📈 TitusAI training progress',
                        training_status_fields(
                            run_name,
                            global_step,
                            tokens_seen,
                            latest_loss,
                            smoothed_loss,
                            last_validation_loss,
                            scheduler.get_last_lr()[0],
                            latest_tokens_per_second,
                            now - training_start,
                            TRAIN_CONFIG['max_train_tokens'],
                            latest_snapshot_name,
                            format_peak_gpu_memory(device),
                        ),
                        description=training_status_description(
                            tokens_seen,
                            TRAIN_CONFIG['max_train_tokens'],
                        ),
                        color='purple',
                    )
                    status_timer = now

                validation_improved = False
                if global_step % TRAIN_CONFIG['validation_interval_steps'] == 0:
                    last_validation_loss = validate(
                        model,
                        validation_loader,
                        device,
                        world_size,
                    )
                    validation_improved = last_validation_loss < best_validation_loss
                    best_validation_loss = min(
                        best_validation_loss,
                        last_validation_loss,
                    )
                    if rank == 0:
                        print(
                            f'[+] validation_loss={last_validation_loss:.4f} '
                            f'best={best_validation_loss:.4f}'
                        )
                        notifier.send(
                            '✅ New best validation loss' if validation_improved else '🧪 Validation complete',
                            [
                                ('Run', run_name),
                                ('Optimizer step', f'{global_step:,}'),
                                ('Validation loss', f'{last_validation_loss:.4f}'),
                                ('Best validation loss', f'{best_validation_loss:.4f}'),
                                ('Current train loss', 'Unavailable' if latest_loss is None else f'{latest_loss:.4f}'),
                                ('Smoothed train loss', 'Unavailable' if smoothed_loss is None else f'{smoothed_loss:.4f}'),
                                ('Tokens processed', f'{tokens_seen:,} / {TRAIN_CONFIG["max_train_tokens"]:,}', False),
                            ],
                            description=(
                                'The model reached a new best validation score.'
                                if validation_improved
                                else training_status_description(
                                    tokens_seen,
                                    TRAIN_CONFIG['max_train_tokens'],
                                )
                            ),
                            color='green' if validation_improved else 'gold',
                        )

                periodic_checkpoint = (
                    global_step % TRAIN_CONFIG['checkpoint_interval_steps'] == 0
                )
                if periodic_checkpoint or validation_improved:
                    save_full_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        model_config,
                        checkpoint_path,
                        rank,
                        world_size,
                        global_step,
                        tokens_seen,
                        best_validation_loss,
                        epoch,
                        samples_seen_in_epoch,
                    )

                if tokens_seen >= TRAIN_CONFIG['max_train_tokens']:
                    token_limit_reached = True
                    break

            if accumulation_step != 0:
                optimizer.zero_grad(set_to_none=True)
                accumulated_local_tokens = 0
                accumulated_local_samples = 0
                accumulation_step = 0

            if token_limit_reached:
                break

            epoch += 1
            samples_seen_in_epoch = 0

        save_full_checkpoint(
            model,
            optimizer,
            scheduler,
            model_config,
            checkpoint_path,
            rank,
            world_size,
            global_step,
            tokens_seen,
            best_validation_loss,
            epoch,
            samples_seen_in_epoch,
        )

        if rank == 0:
            destination = save_inference_snapshot(
                model,
                model_config,
                train_dataset.manifest['tokenizer'],
                snapshot_path,
                TRAIN_CONFIG['snapshot_keep'],
                global_step,
                tokens_seen,
                last_validation_loss,
            )
            latest_snapshot_name = destination.name
            print(f'[+] Final snapshot: {destination.name}')
            notifier.send(
                '🏁 TitusAI training complete',
                training_status_fields(
                    run_name,
                    global_step,
                    tokens_seen,
                    latest_loss,
                    smoothed_loss,
                    last_validation_loss,
                    scheduler.get_last_lr()[0],
                    latest_tokens_per_second,
                    time.monotonic() - training_start,
                    TRAIN_CONFIG['max_train_tokens'],
                    latest_snapshot_name,
                    format_peak_gpu_memory(device),
                ),
                description='Training finished and the final checkpoint is safely stored.',
                color='green',
            )

    except KeyboardInterrupt:
        optimizer.zero_grad(set_to_none=True)
        save_full_checkpoint(
            model,
            optimizer,
            scheduler,
            model_config,
            checkpoint_path,
            rank,
            world_size,
            global_step,
            tokens_seen,
            best_validation_loss,
            epoch,
            samples_seen_in_epoch,
        )
        if rank == 0:
            destination = save_inference_snapshot(
                model,
                model_config,
                train_dataset.manifest['tokenizer'],
                snapshot_path,
                TRAIN_CONFIG['snapshot_keep'],
                global_step,
                tokens_seen,
                last_validation_loss,
            )
            latest_snapshot_name = destination.name
            print(f'\n[+] Interrupted; saved snapshot: {destination.name}')
            notifier.send(
                '⏸️ TitusAI training interrupted safely',
                training_status_fields(
                    run_name,
                    global_step,
                    tokens_seen,
                    latest_loss,
                    smoothed_loss,
                    last_validation_loss,
                    scheduler.get_last_lr()[0],
                    latest_tokens_per_second,
                    time.monotonic() - training_start,
                    TRAIN_CONFIG['max_train_tokens'],
                    latest_snapshot_name,
                    format_peak_gpu_memory(device),
                ),
                description='The current checkpoint and inference snapshot were saved before shutdown.',
                color='orange',
            )

    except Exception as error:
        if rank == 0:
            notifier.send(
                '❌ TitusAI training failed',
                [
                    ('Run', run_name),
                    ('Optimizer step', f'{global_step:,}'),
                    ('Tokens processed', f'{tokens_seen:,}'),
                    ('Error type', type(error).__name__),
                    ('Latest snapshot', latest_snapshot_name, False),
                ],
                description=f'```text\n{str(error)}\n```',
                color='red',
            )
        raise

    finally:
        if notifier is not None:
            notifier.close()
        cleanup_distributed(world_size)


if __name__ == '__main__':
    main()
