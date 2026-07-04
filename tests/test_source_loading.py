import sys
from types import SimpleNamespace

import huggingface_hub

import prepare_data


class FakeDataset:
    def __init__(self):
        self.shuffle_calls = []

    def shuffle(self, seed, buffer_size):
        self.shuffle_calls.append((seed, buffer_size))
        return self


class FakeApi:
    def list_repo_files(self, repo_id, repo_type, revision):
        assert repo_id == 'example/code'
        assert repo_type == 'dataset'
        assert revision == 'refs/convert/parquet'
        return [
            'README.md',
            'stage5-auto-format/train/0001.parquet',
            'stage5-auto-format/train/0000.parquet',
            'stage4-llm-rewrite/train/0000.parquet',
        ]


def fake_hf_hub_url(repo_id, filename, repo_type, revision):
    return f'https://example.test/{revision}/{repo_id}/{filename}'


def test_converted_parquet_files_selects_config_and_split(monkeypatch):
    monkeypatch.setattr(huggingface_hub, 'HfApi', FakeApi)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_url', fake_hf_hub_url)

    source = {
        'name': 'swallowcode',
        'dataset': 'example/code',
        'config': 'stage5-auto-format',
        'split': 'train',
    }

    files = prepare_data.converted_parquet_files(source)

    assert files == [
        'https://example.test/refs/convert/parquet/example/code/'
        'stage5-auto-format/train/0000.parquet',
        'https://example.test/refs/convert/parquet/example/code/'
        'stage5-auto-format/train/0001.parquet',
    ]


def test_source_check_does_not_shuffle(monkeypatch):
    captured = {}
    dataset = FakeDataset()

    def load_dataset(**arguments):
        captured.update(arguments)
        return dataset

    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=load_dataset))

    source = {
        'name': 'math',
        'dataset': 'example/math',
        'config': '4plus',
        'data_dir': None,
        'split': 'train',
        'columns': ['text'],
    }

    result = prepare_data.load_source_stream(source, shuffle=False)

    assert result is dataset
    assert dataset.shuffle_calls == []
    assert captured['columns'] == ['text']
    assert captured['name'] == '4plus'


def test_converted_parquet_loader_reads_only_selected_columns(monkeypatch):
    captured = {}
    dataset = FakeDataset()

    def load_dataset(**arguments):
        captured.update(arguments)
        return dataset

    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=load_dataset))
    monkeypatch.setattr(
        prepare_data,
        'converted_parquet_files',
        lambda source: ['https://example.test/0000.parquet'],
    )

    source = {
        'name': 'swallowcode',
        'dataset': 'example/code',
        'config': 'stage5-auto-format',
        'data_dir': None,
        'split': 'train',
        'loader': 'converted_parquet',
        'columns': ['text'],
    }

    result = prepare_data.load_source_stream(source)

    assert result is dataset
    assert captured['path'] == 'parquet'
    assert captured['columns'] == ['text']
    assert captured['data_files']['train'] == ['https://example.test/0000.parquet']
    assert dataset.shuffle_calls == [(1337, 10_000)]
