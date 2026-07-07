import os

import torch

from checkpoint import (
    find_latest_checkpoint_recursive,
    find_latest_snapshot,
    find_snapshots,
    load_snapshot,
    save_inference_snapshot,
)
from model import TitusModel


def test_snapshot_rotation(tmp_path, tiny_config):
    model = TitusModel(tiny_config)

    for step in range(12):
        save_inference_snapshot(
            model,
            tiny_config,
            {'vocab_size': tiny_config['vocab_size']},
            tmp_path,
            10,
            step,
            step * 100,
        )

    snapshots = list(tmp_path.glob('snapshot_*.pt'))
    assert len(snapshots) == 10
    assert not list(tmp_path.glob('*.writing'))

    latest = find_latest_snapshot(tmp_path)
    data = torch.load(latest, map_location='cpu', weights_only=False)
    assert data['global_step'] == 11


def test_find_latest_snapshot_recursively(tmp_path):
    old_snapshot = tmp_path / 'pretrain' / 'snapshot_09.pt'
    new_snapshot = tmp_path / 'instructions_50m' / 'snapshot_06.pt'
    old_snapshot.parent.mkdir()
    new_snapshot.parent.mkdir()
    old_snapshot.touch()
    new_snapshot.touch()
    os.utime(old_snapshot, (100, 100))
    os.utime(new_snapshot, (200, 200))

    assert find_latest_snapshot(tmp_path) == new_snapshot


def test_find_snapshots_newest_first(tmp_path):
    oldest = tmp_path / 'pretrain' / 'snapshot_09.pt'
    middle = tmp_path / 'instructions_50m' / 'snapshot_06.pt'
    newest = tmp_path / 'conversations_50m' / 'snapshot_02.pt'

    for path in (oldest, middle, newest):
        path.parent.mkdir(exist_ok=True)
        path.touch()

    os.utime(oldest, (100, 100))
    os.utime(middle, (200, 200))
    os.utime(newest, (300, 300))

    assert find_snapshots(tmp_path) == [newest, middle, oldest]


def test_snapshot_round_trip(tmp_path, tiny_config):
    torch.manual_seed(25)
    first_model = TitusModel(tiny_config)
    path = save_inference_snapshot(
        first_model,
        tiny_config,
        {'vocab_size': tiny_config['vocab_size']},
        tmp_path,
        10,
        1,
        100,
    )

    second_model = TitusModel(tiny_config)
    load_snapshot(path, second_model)

    input_ids = torch.randint(0, tiny_config['vocab_size'], (1, 8))
    first_logits = first_model.eval()(input_ids, return_logits=True)['logits']
    second_logits = second_model.eval()(input_ids, return_logits=True)['logits']
    assert torch.allclose(first_logits, second_logits, atol=2e-3)


def test_training_checkpoint_round_trip(tmp_path, tiny_config):
    from checkpoint import load_training_checkpoint, save_training_checkpoint

    torch.manual_seed(31)
    first_model = TitusModel(tiny_config)
    optimizer = torch.optim.AdamW(first_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    input_ids = torch.randint(0, tiny_config['vocab_size'], (1, 8))
    labels = torch.randint(0, tiny_config['vocab_size'], (1, 8))
    loss = first_model(input_ids, labels=labels, return_logits=False)['loss']
    loss.backward()
    optimizer.step()
    scheduler.step()

    path = save_training_checkpoint(
        first_model,
        optimizer,
        scheduler,
        tiny_config,
        {'test': True},
        tmp_path,
        3,
        7,
        4096,
        4.5,
        2,
        123,
        None,
    )

    second_model = TitusModel(tiny_config)
    second_optimizer = torch.optim.AdamW(second_model.parameters(), lr=1e-3)
    second_scheduler = torch.optim.lr_scheduler.LambdaLR(
        second_optimizer,
        lambda step: 1.0,
    )
    checkpoint = load_training_checkpoint(
        path,
        second_model,
        second_optimizer,
        second_scheduler,
    )

    assert checkpoint['global_step'] == 7
    assert checkpoint['samples_seen_in_epoch'] == 123
    for first, second in zip(first_model.parameters(), second_model.parameters()):
        assert torch.equal(first, second)


def test_find_latest_checkpoint_recursively(tmp_path):
    old_checkpoint = tmp_path / 'pretrain' / 'checkpoint_000000100.pt'
    new_checkpoint = tmp_path / 'conversations_50m' / 'checkpoint_000000020.pt'
    old_checkpoint.parent.mkdir()
    new_checkpoint.parent.mkdir()
    old_checkpoint.touch()
    new_checkpoint.touch()
    os.utime(old_checkpoint, (100, 100))
    os.utime(new_checkpoint, (200, 200))

    assert find_latest_checkpoint_recursive(tmp_path) == new_checkpoint
