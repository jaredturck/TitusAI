import os
import time
from pathlib import Path

from config import (
    DATA_SOURCES,
    PREPARE_CONFIG,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
)
from data_utils import (
    DeduplicationStore,
    SequencePacker,
    ShardWriter,
    build_document_id,
    deterministic_split,
    extract_first,
    normalize_document,
    write_manifest,
)
from tokenizer import (
    encode_document,
    get_tokenizer_metadata,
    load_tokenizer,
    save_tokenizer,
)


def clear_prepared_directory(output_path):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for path in output_path.iterdir():
        if path.is_file() and (
            path.name.endswith('.bin')
            or path.name.endswith('.writing')
            or path.name == 'manifest.json'
        ):
            path.unlink()


def converted_parquet_files(source):
    from huggingface_hub import HfApi, hf_hub_url

    revision = 'refs/convert/parquet'
    prefix = f'{source["config"]}/{source["split"]}/'
    api = HfApi()
    files = api.list_repo_files(
        source['dataset'],
        repo_type='dataset',
        revision=revision,
    )
    parquet_files = [
        filename
        for filename in files
        if filename.startswith(prefix) and filename.endswith('.parquet')
    ]

    if not parquet_files:
        raise RuntimeError(
            f'No converted Parquet files found for {source["name"]} '
            f'under {prefix}'
        )

    return [
        hf_hub_url(
            source['dataset'],
            filename,
            repo_type='dataset',
            revision=revision,
        )
        for filename in sorted(parquet_files)
    ]


def load_source_stream(source, shuffle=True):
    from datasets import load_dataset

    if source.get('loader') == 'converted_parquet':
        arguments = {
            'path': 'parquet',
            'data_files': {
                source['split']: converted_parquet_files(source),
            },
            'split': source['split'],
            'streaming': True,
            'columns': source['columns'],
        }
    else:
        arguments = {
            'path': source['dataset'],
            'split': source['split'],
            'streaming': True,
        }

        if source['config'] is not None:
            arguments['name'] = source['config']

        if source['data_dir'] is not None:
            arguments['data_dir'] = source['data_dir']

        if source.get('columns') is not None:
            arguments['columns'] = source['columns']

    dataset = load_dataset(**arguments)
    if not shuffle:
        return dataset

    return dataset.shuffle(
        seed=PREPARE_CONFIG['random_seed'],
        buffer_size=PREPARE_CONFIG['shuffle_buffer_size'],
    )


def source_token_target(source, target_scale):
    return max(1, int(source['target_tokens'] * target_scale))


def create_packers(source_name):
    sequence_length = PREPARE_CONFIG['sequence_length']
    sequences_per_shard = PREPARE_CONFIG['sequences_per_shard']

    train_writer = ShardWriter(
        TRAIN_DATA_PATH,
        source_name,
        sequence_length,
        sequences_per_shard,
        store_loss_mask=False,
    )
    validation_writer = ShardWriter(
        VALIDATION_DATA_PATH,
        source_name,
        sequence_length,
        sequences_per_shard,
        store_loss_mask=False,
    )

    return {
        'train': SequencePacker(train_writer, sequence_length),
        'validation': SequencePacker(validation_writer, sequence_length),
    }


def prepare_source(source, tokenizer, deduplicator, target_scale):
    source_name = source['name']
    target_tokens = source_token_target(source, target_scale)
    packers = create_packers(source_name)
    dataset = load_source_stream(source)
    processed_tokens = 0
    accepted_documents = 0
    rejected_documents = 0
    started = time.monotonic()
    last_report = started

    print(f'[+] Preparing {source_name} to {target_tokens:,} tokens')

    for record in dataset:
        raw_text = extract_first(record, source['text_fields'])
        if not isinstance(raw_text, str):
            rejected_documents += 1
            continue

        text = normalize_document(raw_text)
        if not text.strip() or len(text) < PREPARE_CONFIG['minimum_document_characters']:
            rejected_documents += 1
            continue

        if len(text) > PREPARE_CONFIG['maximum_document_characters']:
            text = text[:PREPARE_CONFIG['maximum_document_characters']]

        if PREPARE_CONFIG['exact_deduplication'] and deduplicator.is_duplicate_document(text):
            rejected_documents += 1
            continue

        paragraph_deduplication = source.get(
            'paragraph_deduplication',
            PREPARE_CONFIG['paragraph_deduplication'],
        )
        if paragraph_deduplication:
            text = deduplicator.remove_duplicate_paragraphs(text)

        if len(text) < PREPARE_CONFIG['minimum_document_characters']:
            rejected_documents += 1
            continue

        document_id = build_document_id(
            record,
            source_name,
            source['id_fields'],
            text,
        )
        split = deterministic_split(
            document_id,
            PREPARE_CONFIG['validation_fraction'],
        )
        token_ids = encode_document(tokenizer, text)
        packers[split].add_document(token_ids)
        processed_tokens += len(token_ids)
        accepted_documents += 1

        now = time.monotonic()
        if now - last_report >= 10:
            elapsed = now - started
            tokens_per_second = int(processed_tokens / max(elapsed, 1))
            print(
                f'[+] {source_name}: {processed_tokens:,}/{target_tokens:,} tokens, '
                f'{accepted_documents:,} documents, {tokens_per_second:,} tokens/s'
            )
            last_report = now

        if processed_tokens >= target_tokens:
            break

    train_statistics = packers['train'].close()
    validation_statistics = packers['validation'].close()

    return {
        'name': source_name,
        'target_tokens': target_tokens,
        'processed_tokens': processed_tokens,
        'accepted_documents': accepted_documents,
        'rejected_documents': rejected_documents,
        'train': train_statistics,
        'validation': validation_statistics,
    }


def main():
    clear_prepared_directory(TRAIN_DATA_PATH)
    clear_prepared_directory(VALIDATION_DATA_PATH)

    database_path = PREPARE_CONFIG['deduplication_database']
    for suffix in ['', '-wal', '-shm']:
        database_file = Path(f'{database_path}{suffix}')
        if database_file.exists():
            database_file.unlink()

    tokenizer = load_tokenizer()
    assert len(tokenizer) < 65_536
    save_tokenizer(tokenizer)
    tokenizer_metadata = get_tokenizer_metadata(tokenizer)

    configured_tokens = sum(source['target_tokens'] for source in DATA_SOURCES)
    maximum_tokens = PREPARE_CONFIG['max_total_tokens']
    target_scale = min(1.0, maximum_tokens / configured_tokens)

    deduplicator = DeduplicationStore(
        database_path,
        PREPARE_CONFIG['paragraph_minimum_characters'],
    )

    source_statistics = []
    train_shards = []
    validation_shards = []

    for source in DATA_SOURCES:
        statistics = prepare_source(
            source,
            tokenizer,
            deduplicator,
            target_scale,
        )
        train_shards.extend(statistics['train']['shards'])
        validation_shards.extend(statistics['validation']['shards'])
        source_statistics.append({
            'name': statistics['name'],
            'target_tokens': statistics['target_tokens'],
            'processed_tokens': statistics['processed_tokens'],
            'accepted_documents': statistics['accepted_documents'],
            'rejected_documents': statistics['rejected_documents'],
            'train': {
                key: value
                for key, value in statistics['train'].items()
                if key != 'shards'
            },
            'validation': {
                key: value
                for key, value in statistics['validation'].items()
                if key != 'shards'
            },
        })

    deduplicator.close()

    train_manifest = write_manifest(
        TRAIN_DATA_PATH,
        tokenizer_metadata,
        PREPARE_CONFIG['sequence_length'],
        source_statistics,
        train_shards,
    )
    validation_manifest = write_manifest(
        VALIDATION_DATA_PATH,
        tokenizer_metadata,
        PREPARE_CONFIG['sequence_length'],
        source_statistics,
        validation_shards,
    )

    print(f'[+] Training sequences: {train_manifest["num_sequences"]:,}')
    print(f'[+] Validation sequences: {validation_manifest["num_sequences"]:,}')
    print('[+] Data preparation complete')


if __name__ == '__main__':
    main()
