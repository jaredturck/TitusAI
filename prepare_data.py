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


def build_manifest(split, source_statistics, tokenizer_metadata):
    sources = []
    total_sequences = 0
    total_tokens = 0

    for source in source_statistics:
        split_statistics = source[split]
        total_sequences += split_statistics['sequences']
        total_tokens += split_statistics['tokens']
        sources.append({
            'name': source['name'],
            'target_tokens': source['target_tokens'],
            'processed_tokens': source['processed_tokens'],
            'accepted_documents': source['accepted_documents'],
            'rejected_documents': source['rejected_documents'],
            'sequences': split_statistics['sequences'],
            'tokens': split_statistics['tokens'],
            'shards': split_statistics['shards'],
        })

    return {
        'format_version': 1,
        'split': split,
        'sequence_length': PREPARE_CONFIG['sequence_length'],
        'token_dtype': 'uint16',
        'segment_dtype': 'uint16',
        'has_loss_mask': False,
        'total_sequences': total_sequences,
        'total_tokens': total_tokens,
        'tokenizer': tokenizer_metadata,
        'sources': sources,
        'preparation': {
            'minimum_document_characters': PREPARE_CONFIG['minimum_document_characters'],
            'maximum_document_characters': PREPARE_CONFIG['maximum_document_characters'],
            'validation_fraction': PREPARE_CONFIG['validation_fraction'],
            'exact_deduplication': PREPARE_CONFIG['exact_deduplication'],
            'paragraph_deduplication': PREPARE_CONFIG['paragraph_deduplication'],
            'random_seed': PREPARE_CONFIG['random_seed'],
        },
    }


def main():
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    TRAIN_DATA_PATH.mkdir(parents=True, exist_ok=True)
    VALIDATION_DATA_PATH.mkdir(parents=True, exist_ok=True)
    clear_prepared_directory(TRAIN_DATA_PATH)
    clear_prepared_directory(VALIDATION_DATA_PATH)

    tokenizer = load_tokenizer()
    save_tokenizer(tokenizer)
    tokenizer_metadata = get_tokenizer_metadata(tokenizer)

    requested_tokens = sum(source['target_tokens'] for source in DATA_SOURCES)
    target_scale = min(1.0, PREPARE_CONFIG['max_total_tokens'] / requested_tokens)
    deduplicator = DeduplicationStore(
        PREPARE_CONFIG['deduplication_database'],
        PREPARE_CONFIG['paragraph_minimum_characters'],
    )
    source_statistics = []

    try:
        for source in DATA_SOURCES:
            statistics = prepare_source(
                source,
                tokenizer,
                deduplicator,
                target_scale,
            )
            source_statistics.append(statistics)
    finally:
        deduplicator.close()

    train_manifest = build_manifest(
        'train',
        source_statistics,
        tokenizer_metadata,
    )
    validation_manifest = build_manifest(
        'validation',
        source_statistics,
        tokenizer_metadata,
    )
    write_manifest(TRAIN_DATA_PATH / 'manifest.json', train_manifest)
    write_manifest(VALIDATION_DATA_PATH / 'manifest.json', validation_manifest)

    print(
        f'[+] Prepared {train_manifest["total_tokens"]:,} training tokens and '
        f'{validation_manifest["total_tokens"]:,} validation tokens'
    )


def run():
    main()
    hard_exit_after_success()


if __name__ == '__main__':
    run()
