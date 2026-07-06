import os

import train


def test_select_training_config_applies_requested_mode(monkeypatch):
    train_config = {'seed': 1337}
    train_configs = {
        'pretrain': {
            'run_name': 'pretrain',
            'isolate_packed_documents': False,
        },
        'instruction': {
            'run_name': 'instructions',
            'isolate_packed_documents': True,
        },
    }
    monkeypatch.setattr(train, 'TRAIN_CONFIG', train_config)
    monkeypatch.setattr(train, 'TRAIN_CONFIGS', train_configs)

    assert train.select_training_config(['instruction'])
    assert train_config == {
        'seed': 1337,
        'run_name': 'instructions',
        'isolate_packed_documents': True,
    }


def test_select_training_config_requires_one_known_mode(monkeypatch, capsys):
    monkeypatch.setattr(train, 'TRAIN_CONFIG', {'seed': 1337})
    monkeypatch.setattr(train, 'TRAIN_CONFIGS', {'pretrain': {}})
    monkeypatch.setenv('RANK', '0')

    assert not train.select_training_config([])
    assert not train.select_training_config(['unknown'])

    output = capsys.readouterr().out
    assert 'Training stopped: provide pretrain or instruction' in output


def test_resolve_initial_weights_uses_newest_snapshot(tmp_path):
    older = tmp_path / 'snapshot_00.pt'
    newer = tmp_path / 'snapshot_01.pt'
    older.touch()
    newer.touch()

    older_time = newer.stat().st_mtime - 10
    os.utime(older, (older_time, older_time))

    assert train.resolve_initial_weights(tmp_path) == newer


def test_resolve_initial_weights_keeps_explicit_file(tmp_path):
    weights_path = tmp_path / 'snapshot_custom.pt'
    weights_path.touch()

    assert train.resolve_initial_weights(weights_path) == weights_path


def test_instruction_config_uses_fast_causal_training():
    from config import TRAIN_CONFIGS

    instruction = TRAIN_CONFIGS['instruction']
    assert instruction['isolate_packed_documents'] is False
    assert instruction['micro_batch_size'] == 4
    assert instruction['gradient_accumulation_steps'] == 8
    assert instruction['gradient_checkpointing'] is False
    assert instruction['max_train_tokens'] == 50_000_000
    assert instruction['learning_rate'] == 3e-5


def test_resolve_initial_weights_prefers_full_checkpoint(tmp_path):
    checkpoint = tmp_path / 'checkpoints' / 'pretrain' / 'checkpoint_000000100.pt'
    snapshot = tmp_path / 'snapshots' / 'pretrain' / 'snapshot_00.pt'
    checkpoint.parent.mkdir(parents=True)
    snapshot.parent.mkdir(parents=True)
    checkpoint.touch()
    snapshot.touch()
    os.utime(checkpoint, (100, 100))
    os.utime(snapshot, (200, 200))

    assert train.resolve_initial_weights(tmp_path) == checkpoint


def test_select_training_start_resumes_only_newest_current_checkpoint(tmp_path):
    current_path = tmp_path / 'checkpoints' / 'conversations_50m'
    other_path = tmp_path / 'checkpoints' / 'pretrain'
    current_path.mkdir(parents=True)
    other_path.mkdir(parents=True)
    current = current_path / 'checkpoint_000000010.pt'
    other = other_path / 'checkpoint_000003818.pt'
    current.touch()
    other.touch()
    os.utime(current, (200, 200))
    os.utime(other, (100, 100))

    resume, initial = train.select_training_start(
        current_path,
        tmp_path,
        True,
    )
    assert resume == current
    assert initial is None

    os.utime(other, (300, 300))
    resume, initial = train.select_training_start(
        current_path,
        tmp_path,
        True,
    )
    assert resume is None
    assert initial == other
