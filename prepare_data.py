import json
import os
import random
import time
from pathlib import Path

from process_utils import hard_exit_after_success

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
    recover_source_shards,
    remove_stale_writing_files,
    write_manifest,
)
from tokenizer import (
    encode_document,
    get_tokenizer_metadata,
    load_tokenizer,
    save_tokenizer,
)


class JsonlTextStream:
    def __init__(self, source):
        self.source = source

    def __iter__(self):
        from huggingface_hub import HfFileSystem

        filesystem = HfFileSystem()
        data_glob = self.source['data_glob'].strip('/')
        pattern = f'datasets/{self.source["dataset"]}/{data_glob}'
        files = sorted(filesystem.glob(pattern))

        if not files:
            raise RuntimeError(
                f'No JSONL files found for {self.source["name"]} under {data_glob}'
            )

        text_field = self.source.get('jsonl_text_field', 'text')
        block_size = self.source.get('read_block_size', 8 * 1024 * 1024)

        for filename in files:
            with filesystem.open(filename, 'rb', block_size=block_size) as file:
                for line_number, line in enumerate(file, start=1):
                    if not line.strip():
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise RuntimeError(
                            f'Invalid JSON in {filename} at line {line_number}'
                        ) from error

                    yield {
                        text_field: record.get(text_field),
                    }


class BufferedShuffleStream:
    def __init__(self, dataset, seed, buffer_size):
        self.dataset = dataset
        self.seed = seed
        self.buffer_size = buffer_size

    def __iter__(self):
        random_generator = random.Random(self.seed)
        iterator = iter(self.dataset)
        buffer = []

        for _ in range(self.buffer_size):
            try:
                buffer.append(next(iterator))
            except StopIteration:
                break

        while buffer:
            index = random_generator.randrange(len(buffer))

            try:
                replacement = next(iterator)
            except StopIteration:
                yield buffer.pop(index)
            else:
                yield buffer[index]
                buffer[index] = replacement


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


def prepare_output_directory(output_path):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    removed = remove_stale_writing_files(output_path)
    manifest_path = output_path / 'manifest.json'

    if manifest_path.exists():
        manifest_path.unlink()

    return removed


def reset_deduplication_database():
    database_path = Path(PREPARE_CONFIG['deduplication_database'])

    for suffix in ['', '-shm', '-wal']:
        path = Path(f'{database_path}{suffix}')
        if path.exists():
            path.unlink()


def load_source_stream(source, shuffle=True):
    from datasets import load_dataset

    if source.get('loader') == 'jsonl_text':
        dataset = JsonlTextStream(source)

        if not shuffle:
            return dataset

        return BufferedShuffleStream(
            dataset,
            PREPARE_CONFIG['random_seed'],
            PREPARE_CONFIG['shuffle_buffer_size'],
        )

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
    if not shuffle or source.get('pre_shuffled', False):
        return dataset

    return dataset.shuffle(
        seed=PREPARE_CONFIG['random_seed'],
        buffer_size=PREPARE_CONFIG['shuffle_buffer_size'],
    )


def source_token_target(source, target_scale):
    return max(1, int(source['target_tokens'] * target_scale))


def split_statistics_from_shards(shards):
    num_sequences = sum(shard['num_sequences'] for shard in shards)
    output_tokens = num_sequences * PREPARE_CONFIG['sequence_length']

    return {
        'documents': 0,
        'input_tokens': output_tokens,
        'output_tokens': output_tokens,
        'discarded_tokens': 0,
        'num_sequences': num_sequences,
        'shards': shards,
    }


def recover_source_statistics(source, target_scale):
    source_name = source['name']
    sequence_length = PREPARE_CONFIG['sequence_length']
    train_shards = recover_source_shards(
        TRAIN_DATA_PATH,
        source_name,
        sequence_length,
        store_loss_mask=False,
    )
    validation_shards = recover_source_shards(
        VALIDATION_DATA_PATH,
        source_name,
        sequence_length,
        store_loss_mask=False,
    )
    train_statistics = split_statistics_from_shards(train_shards)
    validation_statistics = split_statistics_from_shards(validation_shards)
    processed_tokens = (
        train_statistics['output_tokens']
        + validation_statistics['output_tokens']
    )

    return {
        'name': source_name,
        'target_tokens': source_token_target(source, target_scale),
        'processed_tokens': processed_tokens,
        'accepted_documents': 0,
        'rejected_documents': 0,
        'resumed': processed_tokens > 0,
        'train': train_statistics,
        'validation': validation_statistics,
    }


def source_is_complete(statistics):
    tolerance = PREPARE_CONFIG['sequence_length'] * 2
    minimum_tokens = max(0, statistics['target_tokens'] - tolerance)
    return statistics['processed_tokens'] >= minimum_tokens


def create_packers(source_name, existing_statistics):
    sequence_length = PREPARE_CONFIG['sequence_length']
    sequences_per_shard = PREPARE_CONFIG['sequences_per_shard']

    train_writer = ShardWriter(
        TRAIN_DATA_PATH,
        source_name,
        sequence_length,
        sequences_per_shard,
        store_loss_mask=False,
        existing_shards=existing_statistics['train']['shards'],
    )
    validation_writer = ShardWriter(
        VALIDATION_DATA_PATH,
        source_name,
        sequence_length,
        sequences_per_shard,
        store_loss_mask=False,
        existing_shards=existing_statistics['validation']['shards'],
    )

    return {
        'train': SequencePacker(train_writer, sequence_length),
        'validation': SequencePacker(validation_writer, sequence_length),
    }


def finalize_split_statistics(statistics):
    statistics['num_sequences'] = sum(
        shard['num_sequences']
        for shard in statistics['shards']
    )
    statistics['output_tokens'] = (
        statistics['num_sequences']
        * PREPARE_CONFIG['sequence_length']
    )
    return statistics


def prepare_source(source, tokenizer, deduplicator, target_scale, existing_statistics):
    source_name = source['name']
    target_tokens = source_token_target(source, target_scale)
    packers = create_packers(source_name, existing_statistics)
    dataset = load_source_stream(source)
    processed_tokens = existing_statistics['processed_tokens']
    accepted_documents = 0
    rejected_documents = 0
    started = time.monotonic()
    last_report = started

    print(
        f'[+] Resuming {source_name} at {processed_tokens:,}/{target_tokens:,} tokens'
    )

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
            new_tokens = processed_tokens - existing_statistics['processed_tokens']
            tokens_per_second = int(new_tokens / max(elapsed, 1))
            print(
                f'[+] {source_name}: {processed_tokens:,}/{target_tokens:,} tokens, '
                f'{accepted_documents:,} new documents, {tokens_per_second:,} tokens/s'
            )
            last_report = now

        if processed_tokens >= target_tokens:
            break

    train_statistics = finalize_split_statistics(packers['train'].close())
    validation_statistics = finalize_split_statistics(packers['validation'].close())
    recovered_documents = (
        existing_statistics['accepted_documents']
        if existing_statistics['accepted_documents'] is not None
        else 0
    )

    return {
        'name': source_name,
        'target_tokens': target_tokens,
        'processed_tokens': (
            train_statistics['output_tokens']
            + validation_statistics['output_tokens']
        ),
        'accepted_documents': recovered_documents + accepted_documents,
        'rejected_documents': rejected_documents,
        'resumed': existing_statistics['processed_tokens'] > 0,
        'train': train_statistics,
        'validation': validation_statistics,
    }


def manifest_source_statistics(source_statistics, split):
    sources = []

    for source in source_statistics:
        split_statistics = source[split]
        sources.append({
            'name': source['name'],
            'target_tokens': source['target_tokens'],
            'processed_tokens': source['processed_tokens'],
            'accepted_documents': source['accepted_documents'],
            'rejected_documents': source['rejected_documents'],
            'resumed': source['resumed'],
            'num_sequences': split_statistics['num_sequences'],
            'output_tokens': split_statistics['output_tokens'],
        })

    return sources


def write_dataset_manifests(source_statistics, tokenizer_metadata):
    train_shards = []
    validation_shards = []

    for source in source_statistics:
        train_shards.extend(source['train']['shards'])
        validation_shards.extend(source['validation']['shards'])

    train_manifest = write_manifest(
        TRAIN_DATA_PATH,
        tokenizer_metadata,
        PREPARE_CONFIG['sequence_length'],
        manifest_source_statistics(source_statistics, 'train'),
        train_shards,
    )
    validation_manifest = write_manifest(
        VALIDATION_DATA_PATH,
        tokenizer_metadata,
        PREPARE_CONFIG['sequence_length'],
        manifest_source_statistics(source_statistics, 'validation'),
        validation_shards,
    )

    return train_manifest, validation_manifest


def write_preparation_state(source_statistics):
    state_path = TRAIN_DATA_PATH.parent / 'preparation_state.json'
    state = {
        'sequence_length': PREPARE_CONFIG['sequence_length'],
        'max_total_tokens': PREPARE_CONFIG['max_total_tokens'],
        'sources': [
            {
                'name': source['name'],
                'target_tokens': source['target_tokens'],
                'processed_tokens': source['processed_tokens'],
                'complete': source_is_complete(source),
            }
            for source in source_statistics
        ],
    }
    temporary_path = Path(f'{state_path}.writing')
    temporary_path.write_text(json.dumps(state, indent=2), encoding='utf-8')
    os.replace(temporary_path, state_path)


def main():
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    removed_train = prepare_output_directory(TRAIN_DATA_PATH)
    removed_validation = prepare_output_directory(VALIDATION_DATA_PATH)

    for filename in removed_train + removed_validation:
        print(f'[+] Removed incomplete temporary file: {filename}')

    tokenizer = load_tokenizer()
    save_tokenizer(tokenizer)
    tokenizer_metadata = get_tokenizer_metadata(tokenizer)

    requested_tokens = sum(source['target_tokens'] for source in DATA_SOURCES)
    target_scale = min(1.0, PREPARE_CONFIG['max_total_tokens'] / requested_tokens)
    recovered_statistics = [
        recover_source_statistics(source, target_scale)
        for source in DATA_SOURCES
    ]
    has_existing_data = any(
        source['processed_tokens'] > 0
        for source in recovered_statistics
    )
    incomplete_sources = [
        source
        for source in recovered_statistics
        if not source_is_complete(source)
    ]

    if has_existing_data:
        recovered_tokens = sum(
            source['processed_tokens']
            for source in recovered_statistics
        )
        print(
            f'[+] Found {recovered_tokens:,} existing prepared tokens; '
            'validating and resuming instead of starting over'
        )
    else:
        reset_deduplication_database()
        print('[+] No existing prepared data found; starting a fresh run')

    if has_existing_data and incomplete_sources:
        database_path = Path(PREPARE_CONFIG['deduplication_database'])
        if not database_path.exists():
            raise RuntimeError(
                'Prepared shards exist but the deduplication database is missing. '
                'Cannot safely resume a partial run without duplicating documents.'
            )

    if not incomplete_sources:
        train_manifest, validation_manifest = write_dataset_manifests(
            recovered_statistics,
            tokenizer_metadata,
        )
        write_preparation_state(recovered_statistics)
        train_tokens = train_manifest['num_sequences'] * PREPARE_CONFIG['sequence_length']
        validation_tokens = (
            validation_manifest['num_sequences']
            * PREPARE_CONFIG['sequence_length']
        )
        print('[+] Existing shards already satisfy every source target')
        print(
            f'[+] Recovered {train_tokens:,} training tokens and '
            f'{validation_tokens:,} validation tokens'
        )
        return

    deduplicator = DeduplicationStore(
        PREPARE_CONFIG['deduplication_database'],
        PREPARE_CONFIG['paragraph_minimum_characters'],
    )
    source_statistics = []

    try:
        for source, existing_statistics in zip(DATA_SOURCES, recovered_statistics):
            if source_is_complete(existing_statistics):
                print(
                    f'[+] {source["name"]} already complete at '
                    f'{existing_statistics["processed_tokens"]:,} tokens; skipping'
                )
                statistics = existing_statistics
            else:
                statistics = prepare_source(
                    source,
                    tokenizer,
                    deduplicator,
                    target_scale,
                    existing_statistics,
                )

            source_statistics.append(statistics)
            write_preparation_state(source_statistics)
    finally:
        deduplicator.close()

    missing_sources = [
        source['name']
        for source in source_statistics
        if not source_is_complete(source)
    ]
    if missing_sources:
        raise RuntimeError(
            'Dataset stream ended before these targets were reached: '
            + ', '.join(missing_sources)
        )

    train_manifest, validation_manifest = write_dataset_manifests(
        source_statistics,
        tokenizer_metadata,
    )
    write_preparation_state(source_statistics)
    train_tokens = train_manifest['num_sequences'] * PREPARE_CONFIG['sequence_length']
    validation_tokens = (
        validation_manifest['num_sequences']
        * PREPARE_CONFIG['sequence_length']
    )

    print(
        f'[+] Prepared {train_tokens:,} training tokens and '
        f'{validation_tokens:,} validation tokens'
    )


def run():
    main()
    hard_exit_after_success()


if __name__ == '__main__':
    run()
