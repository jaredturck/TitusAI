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
