import datetime
import os
import random
from pathlib import Path

import numpy as np
import torch

def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model

def state_dict_to_cpu(model, dtype=None):
    state_dict = {}
    for name, tensor in unwrap_model(model).state_dict().items():
        tensor = tensor.detach().cpu()
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype)
        state_dict[name] = tensor
    return state_dict

def atomic_torch_save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(f'{path}.writing')
    torch.save(data, temporary_path)
    os.replace(temporary_path, path)

def select_snapshot_path(snapshot_path, keep):
    snapshot_path = Path(snapshot_path)
    snapshot_path.mkdir(parents=True, exist_ok=True)
    candidates = [snapshot_path / f'snapshot_{index:02d}.pt' for index in range(keep)]

    for candidate in candidates:
        if not candidate.exists():
            return candidate

    return min(candidates, key=lambda path: path.stat().st_mtime)

def save_inference_snapshot(model, model_config, tokenizer_metadata, snapshot_path, keep, global_step, tokens_seen, validation_loss=None):
    destination = select_snapshot_path(snapshot_path, keep)
    snapshot = {
        'format': 'titus_inference_snapshot_v1',
        'model': state_dict_to_cpu(model, dtype=torch.bfloat16),
        'model_config': dict(model_config),
        'tokenizer': tokenizer_metadata,
        'global_step': global_step,
        'tokens_seen': tokens_seen,
        'validation_loss': validation_loss,
        'saved_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    atomic_torch_save(snapshot, destination)
    return destination

def find_snapshots(snapshot_path):
    snapshot_path = Path(snapshot_path)
    snapshots = list(snapshot_path.rglob('snapshot_*.pt'))
    snapshots = [path for path in snapshots if not path.name.endswith('.writing')]
    return sorted(snapshots, key=os.path.getmtime, reverse=True)

def find_latest_snapshot(snapshot_path):
    snapshots = find_snapshots(snapshot_path)
    if not snapshots:
        return None
    return snapshots[0]

def capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state

def restore_rng_state(state):
    if state is None:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])

def save_training_checkpoint(model, optimizer, scheduler, model_config, train_config, checkpoint_path, keep, global_step, tokens_seen, best_validation_loss, epoch, samples_seen_in_epoch, rng_states=None):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_path / f'checkpoint_{global_step:09d}.pt'
    checkpoint = {
        'format': 'titus_training_checkpoint_v1',
        'model': state_dict_to_cpu(model),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'model_config': dict(model_config),
        'train_config': dict(train_config),
        'global_step': global_step,
        'tokens_seen': tokens_seen,
        'best_validation_loss': best_validation_loss,
        'epoch': epoch,
        'samples_seen_in_epoch': samples_seen_in_epoch,
        'rng_states': rng_states,
        'saved_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    atomic_torch_save(checkpoint, destination)

    checkpoints = sorted(
        checkpoint_path.glob('checkpoint_*.pt'),
        key=lambda path: path.stat().st_mtime,
    )
    for old_checkpoint in checkpoints[:-keep]:
        old_checkpoint.unlink()

    return destination

def find_latest_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoints = list(checkpoint_path.glob('checkpoint_*.pt'))
    checkpoints = [path for path in checkpoints if not path.name.endswith('.writing')]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: path.stat().st_mtime)

def find_latest_checkpoint_recursive(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    checkpoints = list(checkpoint_path.rglob('checkpoint_*.pt'))
    checkpoints = [
        path for path in checkpoints
        if not path.name.endswith('.writing')
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

def load_training_checkpoint(path, model, optimizer=None, scheduler=None, rank=0):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    unwrap_model(model).load_state_dict(checkpoint['model'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    rng_states = checkpoint.get('rng_states')
    if isinstance(rng_states, list) and rank < len(rng_states):
        restore_rng_state(rng_states[rank])
    elif isinstance(rng_states, dict):
        restore_rng_state(rng_states)

    return checkpoint

def load_snapshot(path, model):
    snapshot = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(snapshot['model'])
    return snapshot
