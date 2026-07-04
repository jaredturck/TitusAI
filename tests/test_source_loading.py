import io
import json
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


class FakeFileSystem:
    files = {
        'datasets/example/code/stage5-auto-format/python/medium/train_0000.jsonl': [
            {
                'text': 'first program',
                'lint_report': None,
            },
            {
                'text': 'second program',
                'lint_report': {
                    'type': 'syntax_error',
                    'message': 'example',
                },
            },
        ],
    }

    def glob(self, pattern):
        assert pattern == (
            'datasets/example/code/'
            'stage5-auto-format/python/medium/train*.jsonl'
        )
        return sorted(self.files)

    def open(self, filename, mode, block_size):
        assert mode == 'rb'
        assert block_size == 8 * 1024 * 1024
        content = b''.join(
            json.dumps(record).encode('utf-8') + b'\n'
            for record in self.files[filename]
        )
        return io.BytesIO(content)


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


def test_jsonl_text_stream_reads_only_text(monkeypatch):
    monkeypatch.setattr(huggingface_hub, 'HfFileSystem', FakeFileSystem)

    source = {
        'name': 'swallowcode',
        'dataset': 'example/code',
        'loader': 'jsonl_text',
        'data_glob': 'stage5-auto-format/python/medium/train*.jsonl',
        'jsonl_text_field': 'text',
    }

    records = list(prepare_data.load_source_stream(source, shuffle=False))

    assert records == [
        {'text': 'first program'},
        {'text': 'second program'},
    ]


def test_jsonl_text_stream_ignores_inconsistent_metadata(monkeypatch):
    monkeypatch.setattr(huggingface_hub, 'HfFileSystem', FakeFileSystem)

    source = {
        'name': 'swallowcode',
        'dataset': 'example/code',
        'loader': 'jsonl_text',
        'data_glob': 'stage5-auto-format/python/medium/train*.jsonl',
        'jsonl_text_field': 'text',
    }

    iterator = iter(prepare_data.load_source_stream(source, shuffle=False))
    first = next(iterator)
    second = next(iterator)
    iterator.close()

    assert first['text'] == 'first program'
    assert second['text'] == 'second program'


def test_buffered_shuffle_stream_is_deterministic():
    records = [{'text': str(index)} for index in range(20)]

    first = list(prepare_data.BufferedShuffleStream(records, 1337, 5))
    second = list(prepare_data.BufferedShuffleStream(records, 1337, 5))

    assert first == second
    assert sorted(int(record['text']) for record in first) == list(range(20))
    assert first != records


def test_setup_check_closes_source_iterator(monkeypatch):
    import check_setup

    class CloseTrackingIterator:
        def __init__(self):
            self.closed = False
            self.returned = False

        def __iter__(self):
            return self

        def __next__(self):
            if self.returned:
                raise StopIteration

            self.returned = True
            return {'text': 'example document'}

        def close(self):
            self.closed = True

    iterator = CloseTrackingIterator()

    class CloseTrackingDataset:
        def __iter__(self):
            return iterator

    monkeypatch.setattr(check_setup, 'DATA_SOURCES', [{
        'name': 'example',
        'text_fields': ['text'],
    }])
    monkeypatch.setattr(
        check_setup,
        'load_source_stream',
        lambda source, shuffle=False: CloseTrackingDataset(),
    )

    check_setup.check_datasets()

    assert iterator.closed
