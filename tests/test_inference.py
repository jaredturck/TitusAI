import os

import inference


def make_snapshot(path, modified_time):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    os.utime(path, (modified_time, modified_time))
    return path


def test_select_snapshot(monkeypatch, tmp_path, capsys):
    oldest = make_snapshot(
        tmp_path / 'pretrain' / 'snapshot_09.pt',
        100,
    )
    newest = make_snapshot(
        tmp_path / 'conversations_50m' / 'snapshot_02.pt',
        200,
    )

    monkeypatch.setattr(inference, 'SNAPSHOT_PATH', tmp_path)
    monkeypatch.setattr('builtins.input', lambda prompt: '2')

    selected = inference.select_snapshot(newest)

    assert selected == oldest
    output = capsys.readouterr().out
    assert 'conversations_50m/snapshot_02.pt' in output
    assert 'pretrain/snapshot_09.pt' in output
    assert '[current]' in output


def test_select_snapshot_can_cancel(monkeypatch, tmp_path, capsys):
    current = make_snapshot(
        tmp_path / 'conversations_50m' / 'snapshot_02.pt',
        200,
    )

    monkeypatch.setattr(inference, 'SNAPSHOT_PATH', tmp_path)
    monkeypatch.setattr('builtins.input', lambda prompt: '')

    assert inference.select_snapshot(current) is None
    assert 'Snapshot selection cancelled' in capsys.readouterr().out


def test_select_snapshot_rejects_invalid_number(monkeypatch, tmp_path, capsys):
    current = make_snapshot(
        tmp_path / 'conversations_50m' / 'snapshot_02.pt',
        200,
    )

    monkeypatch.setattr(inference, 'SNAPSHOT_PATH', tmp_path)
    monkeypatch.setattr('builtins.input', lambda prompt: '9')

    assert inference.select_snapshot(current) is None
    assert 'Invalid snapshot selection' in capsys.readouterr().out
