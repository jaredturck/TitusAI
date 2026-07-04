import json

import numpy as np
import pytest

import prepare_data
from data_utils import (
    SequencePacker,
    ShardWriter,
    recover_source_shards,
)


def create_pretraining_shard(output_path, source_name, sequence_length, values):
    writer = ShardWriter(
        output_path,
        source_name,
        sequence_length,
        8,
        store_loss_mask=False,
    )
    packer = SequencePacker(writer, sequence_length)
    packer.add_document(values)
    return packer.close()


def test_recover_source_shards_from_existing_files(tmp_path):
    output_path = tmp_path / 'train'
    create_pretraining_shard(output_path, 'example', 4, [1, 2, 3, 4, 5])

    shards = recover_source_shards(
        output_path,
        'example',
        4,
        store_loss_mask=False,
    )

    assert shards == [{
        'source': 'example',
        'tokens': 'example_000000.tokens.bin',
        'segments': 'example_000000.segments.bin',
        'loss_mask': None,
        'num_sequences': 1,
    }]


def test_recover_source_shards_rejects_mismatched_pairs(tmp_path):
    output_path = tmp_path / 'train'
    output_path.mkdir()
    np.arange(5, dtype=np.uint16).tofile(
        output_path / 'example_000000.tokens.bin'
    )

    with pytest.raises(RuntimeError, match='Incomplete shard'):
        recover_source_shards(output_path, 'example', 4)


def test_shard_writer_appends_after_recovered_shards(tmp_path):
    output_path = tmp_path / 'train'
    first_statistics = create_pretraining_shard(
        output_path,
        'example',
        4,
        [1, 2, 3, 4, 5],
    )
    existing_shards = recover_source_shards(output_path, 'example', 4)
    writer = ShardWriter(
        output_path,
        'example',
        4,
        8,
        store_loss_mask=False,
        existing_shards=existing_shards,
    )
    packer = SequencePacker(writer, 4)
    packer.add_document([6, 7, 8, 9, 10])
    second_statistics = packer.close()

    assert first_statistics['num_sequences'] == 1
    assert second_statistics['num_sequences'] == 2
    assert (output_path / 'example_000000.tokens.bin').exists()
    assert (output_path / 'example_000001.tokens.bin').exists()


def test_prepare_data_recovers_complete_run_without_streaming(tmp_path, monkeypatch):
    train_path = tmp_path / 'processed' / 'train'
    validation_path = tmp_path / 'processed' / 'validation'
    create_pretraining_shard(
        train_path,
        'example',
        4,
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    validation_path.mkdir(parents=True)

    source = {
        'name': 'example',
        'dataset': 'unused',
        'config': None,
        'data_dir': None,
        'split': 'train',
        'text_fields': ['text'],
        'id_fields': [],
        'columns': ['text'],
        'target_tokens': 8,
    }
    prepare_config = dict(prepare_data.PREPARE_CONFIG)
    prepare_config.update({
        'sequence_length': 4,
        'sequences_per_shard': 8,
        'max_total_tokens': 8,
        'deduplication_database': tmp_path / 'deduplication.sqlite3',
    })

    monkeypatch.setattr(prepare_data, 'TRAIN_DATA_PATH', train_path)
    monkeypatch.setattr(prepare_data, 'VALIDATION_DATA_PATH', validation_path)
    monkeypatch.setattr(prepare_data, 'DATA_SOURCES', [source])
    monkeypatch.setattr(prepare_data, 'PREPARE_CONFIG', prepare_config)
    monkeypatch.setattr(prepare_data, 'load_tokenizer', lambda: object())
    monkeypatch.setattr(prepare_data, 'save_tokenizer', lambda tokenizer: None)
    monkeypatch.setattr(
        prepare_data,
        'get_tokenizer_metadata',
        lambda tokenizer: {'name': 'test', 'vocab_size': 128},
    )
    monkeypatch.setattr(
        prepare_data,
        'load_source_stream',
        lambda source: pytest.fail('completed source was streamed again'),
    )

    prepare_data.main()

    train_manifest = json.loads(
        (train_path / 'manifest.json').read_text(encoding='utf-8')
    )
    validation_manifest = json.loads(
        (validation_path / 'manifest.json').read_text(encoding='utf-8')
    )
    assert train_manifest['num_sequences'] == 2
    assert validation_manifest['num_sequences'] == 0
    assert train_manifest['shards'][0]['source'] == 'example'
