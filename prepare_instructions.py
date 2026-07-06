import json
import re
import time
from pathlib import Path

from config import INSTRUCTION_CONFIG, INSTRUCTION_SOURCES
from data_utils import (
    SequencePacker,
    ShardWriter,
    deterministic_split,
    stable_hash,
    write_manifest,
)
from prepare_data import BufferedShuffleStream, clear_prepared_directory
from tokenizer import document_end_token_id, get_tokenizer_metadata, load_tokenizer, save_tokenizer


WHITESPACE = re.compile(r'\s+')


def load_instruction_stream(source, pass_index=0, shuffle=True):
    from datasets import load_dataset

    arguments = {
        'path': source['dataset'],
        'split': source['split'],
        'streaming': True,
    }

    if source.get('config') is not None:
        arguments['name'] = source['config']

    dataset = load_dataset(**arguments)

    if not shuffle:
        return dataset

    return BufferedShuffleStream(
        dataset,
        INSTRUCTION_CONFIG['random_seed'] + pass_index,
        INSTRUCTION_CONFIG['shuffle_buffer_size'],
    )


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)

    if hours:
        return f'{hours}h {minutes:02d}m {seconds:02d}s'
    return f'{minutes}m {seconds:02d}s'


def format_instruction_progress(source_name, processed, accepted, rejected, tokens, target_tokens, elapsed):
    conversations_per_second = processed / elapsed if elapsed > 0 else 0
    percentage = 100 * tokens / target_tokens
    progress = (
        f'[+] {source_name}: {tokens:,} / {target_tokens:,} tokens '
        f'({percentage:.2f}%) | processed={processed:,} '
        f'accepted={accepted:,} rejected={rejected:,} '
        f'| {conversations_per_second:,.1f} conversations/s '
        f'| elapsed={format_duration(elapsed)}'
    )

    if conversations_per_second > 0 and tokens > 0:
        tokens_per_conversation = tokens / accepted if accepted > 0 else 0
        if tokens_per_conversation > 0:
            remaining = max(0, target_tokens - tokens)
            eta = remaining / (conversations_per_second * tokens_per_conversation)
            progress += f' | ETA={format_duration(eta)}'

    return progress


def normalize_message(message):
    if not isinstance(message, str):
        return None

    message = message.replace('\r\n', '\n').replace('\r', '\n')
    message = WHITESPACE.sub(' ', message).strip()
    return message or None


def unwrap_conversation_record(record, messages_field):
    if messages_field in record:
        return record

    if len(record) == 1:
        nested = next(iter(record.values()))
        if isinstance(nested, dict) and messages_field in nested:
            return nested

    return record


def extract_conversation_messages(record, source):
    if not isinstance(record, dict):
        return []

    messages_field = source['messages_field']
    record = unwrap_conversation_record(record, messages_field)
    messages = record.get(messages_field)
    if not isinstance(messages, list):
        return []

    text_field = source.get('message_text_field')
    normalized = []

    for message in messages:
        if isinstance(message, str):
            text = message
        elif isinstance(message, dict):
            if text_field is not None:
                text = message.get(text_field)
            else:
                text = None
                for field in ('message', 'utterance', 'value', 'content'):
                    if isinstance(message.get(field), str):
                        text = message[field]
                        break
        else:
            text = None

        text = normalize_message(text)
        if text is not None:
            normalized.append(text)

    return normalized


def build_instruction_tokens(tokenizer, messages):
    conversation = '\n'.join(messages)
    token_ids = tokenizer.encode(conversation, add_special_tokens=False)
    token_ids.append(document_end_token_id(tokenizer))
    return token_ids


def create_source_packers(output_path, source_name):
    sequence_length = INSTRUCTION_CONFIG['sequence_length']
    sequences_per_shard = INSTRUCTION_CONFIG['sequences_per_shard']

    return {
        'train': SequencePacker(
            ShardWriter(
                output_path / 'train',
                source_name,
                sequence_length,
                sequences_per_shard,
                store_loss_mask=False,
            ),
            sequence_length,
        ),
        'validation': SequencePacker(
            ShardWriter(
                output_path / 'validation',
                source_name,
                sequence_length,
                sequences_per_shard,
                store_loss_mask=False,
            ),
            sequence_length,
        ),
    }


def prepare_source(tokenizer, output_path, source):
    packers = create_source_packers(output_path, source['name'])
    target_tokens = source['target_tokens']
    processed_tokens = 0
    processed = 0
    accepted = 0
    rejected = 0
    pass_index = 0
    started = time.monotonic()
    last_log_time = started

    while processed_tokens < target_tokens:
        accepted_before_pass = accepted
        dataset = load_instruction_stream(source, pass_index)

        for record in dataset:
            processed += 1
            messages = extract_conversation_messages(record, source)
            if len(messages) < INSTRUCTION_CONFIG['minimum_messages']:
                rejected += 1
                continue

            token_ids = build_instruction_tokens(tokenizer, messages)
            serialized = json.dumps(messages, ensure_ascii=False)
            document_id = f'{source["name"]}:{stable_hash(serialized)}'
            split = deterministic_split(
                document_id,
                INSTRUCTION_CONFIG['validation_fraction'],
            )
            packers[split].add_document(token_ids)
            processed_tokens += len(token_ids)
            accepted += 1

            if processed % INSTRUCTION_CONFIG['progress_check_conversations'] == 0:
                now = time.monotonic()
                if now - last_log_time >= INSTRUCTION_CONFIG['progress_interval_seconds']:
                    print(
                        format_instruction_progress(
                            source['name'],
                            processed,
                            accepted,
                            rejected,
                            processed_tokens,
                            target_tokens,
                            now - started,
                        ),
                        flush=True,
                    )
                    last_log_time = now

            if processed_tokens >= target_tokens:
                break

        if accepted == accepted_before_pass:
            raise RuntimeError(
                f'No usable conversations found for {source["name"]}'
            )

        pass_index += 1

    print(
        format_instruction_progress(
            source['name'],
            processed,
            accepted,
            rejected,
            processed_tokens,
            target_tokens,
            time.monotonic() - started,
        ),
        flush=True,
    )

    train_statistics = packers['train'].close()
    validation_statistics = packers['validation'].close()
    return {
        'name': source['name'],
        'dataset': source['dataset'],
        'target_tokens': target_tokens,
        'processed_tokens': processed_tokens,
        'accepted_conversations': accepted,
        'rejected_conversations': rejected,
        'passes': pass_index,
        'train': train_statistics,
        'validation': validation_statistics,
    }


def manifest_source_statistics(statistics, split):
    split_statistics = statistics[split]
    return {
        'name': statistics['name'],
        'dataset': statistics['dataset'],
        'target_tokens': statistics['target_tokens'],
        'processed_tokens': statistics['processed_tokens'],
        'accepted_conversations': statistics['accepted_conversations'],
        'rejected_conversations': statistics['rejected_conversations'],
        'passes': statistics['passes'],
        'num_sequences': split_statistics['num_sequences'],
        'output_tokens': split_statistics['output_tokens'],
    }


def main():
    configured_tokens = sum(source['target_tokens'] for source in INSTRUCTION_SOURCES)
    assert configured_tokens == INSTRUCTION_CONFIG['max_total_tokens']

    tokenizer = load_tokenizer()
    save_tokenizer(tokenizer)
    tokenizer_metadata = get_tokenizer_metadata(tokenizer)
    output_path = Path(INSTRUCTION_CONFIG['output_path'])
    train_path = output_path / 'train'
    validation_path = output_path / 'validation'
    clear_prepared_directory(train_path)
    clear_prepared_directory(validation_path)

    source_statistics = [
        prepare_source(tokenizer, output_path, source)
        for source in INSTRUCTION_SOURCES
    ]
    train_shards = []
    validation_shards = []

    for statistics in source_statistics:
        train_shards.extend(statistics['train']['shards'])
        validation_shards.extend(statistics['validation']['shards'])

    write_manifest(
        train_path,
        tokenizer_metadata,
        INSTRUCTION_CONFIG['sequence_length'],
        [
            manifest_source_statistics(statistics, 'train')
            for statistics in source_statistics
        ],
        train_shards,
    )
    write_manifest(
        validation_path,
        tokenizer_metadata,
        INSTRUCTION_CONFIG['sequence_length'],
        [
            manifest_source_statistics(statistics, 'validation')
            for statistics in source_statistics
        ],
        validation_shards,
    )

    print('[+] Conversation data preparation complete')


if __name__ == '__main__':
    main()
