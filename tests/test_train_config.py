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
