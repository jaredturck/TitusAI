import json
import time
from pathlib import Path

from config import INSTRUCTION_CONFIG
from data_utils import (
    SequencePacker,
    ShardWriter,
    deterministic_split,
    stable_hash,
    write_manifest,
)
from prepare_data import clear_prepared_directory
from tokenizer import document_end_token_id, get_tokenizer_metadata, load_tokenizer, save_tokenizer


def load_instruction_stream():
    from datasets import load_dataset

    arguments = {
        'path': INSTRUCTION_CONFIG['dataset'],
        'split': INSTRUCTION_CONFIG['split'],
        'streaming': True,
    }

    if INSTRUCTION_CONFIG['config'] is not None:
        arguments['name'] = INSTRUCTION_CONFIG['config']

    return load_dataset(**arguments)


def instruction_total_conversations(dataset):
    splits = dataset.info.splits
    if splits is None:
        return None

    split = splits.get(INSTRUCTION_CONFIG['split'])
    if split is None:
        return None
    return split.num_examples


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)

    if hours:
        return f'{hours}h {minutes:02d}m {seconds:02d}s'
    return f'{minutes}m {seconds:02d}s'


def format_instruction_progress(processed, total, accepted, rejected, elapsed):
    conversations_per_second = processed / elapsed if elapsed > 0 else 0
    progress = f'[+] Processed {processed:,}'

    if total is not None:
        percentage = 100 * processed / total
        progress += f' / {total:,} conversations ({percentage:.2f}%)'
    else:
        progress += ' conversations'

    progress += (
        f' | accepted={accepted:,} rejected={rejected:,}'
        f' | {conversations_per_second:,.1f} conversations/s'
        f' | elapsed={format_duration(elapsed)}'
    )

    if total is not None and conversations_per_second > 0:
        remaining = max(0, total - processed)
        progress += f' | ETA={format_duration(remaining / conversations_per_second)}'

    return progress


def encode_text(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def build_instruction_tokens(tokenizer, messages):
    token_ids = []
    loss_mask = []
    has_assistant = False

    direct_tokens = encode_text(tokenizer, '<|direct|>\n')
    token_ids.extend(direct_tokens)
    loss_mask.extend([0] * len(direct_tokens))

    for message in messages:
        role = message.get('role')
        content = message.get('content')
        if role not in {'system', 'user', 'assistant'}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue

        prefix = encode_text(tokenizer, f'<|im_start|>{role}\n')
        body = encode_text(tokenizer, content.strip())
        suffix = encode_text(tokenizer, '<|im_end|>\n')

        if role == 'assistant':
            has_assistant = True

        token_ids.extend(prefix)
        loss_mask.extend([0] * len(prefix))
        token_ids.extend(body)
        loss_mask.extend([1 if role == 'assistant' else 0] * len(body))
        token_ids.extend(suffix)
        loss_mask.extend([1 if role == 'assistant' else 0] * len(suffix))

    token_ids.append(document_end_token_id(tokenizer))
    loss_mask.append(1 if has_assistant else 0)
    return token_ids, loss_mask


def create_packers(output_path):
    train_path = output_path / 'train'
    validation_path = output_path / 'validation'
    clear_prepared_directory(train_path)
    clear_prepared_directory(validation_path)

    sequence_length = INSTRUCTION_CONFIG['sequence_length']
    sequences_per_shard = INSTRUCTION_CONFIG['sequences_per_shard']

    return {
        'train_path': train_path,
        'validation_path': validation_path,
        'train': SequencePacker(
            ShardWriter(
                train_path,
                'smol_smoltalk',
                sequence_length,
                sequences_per_shard,
            ),
            sequence_length,
        ),
        'validation': SequencePacker(
            ShardWriter(
                validation_path,
                'smol_smoltalk',
                sequence_length,
                sequences_per_shard,
            ),
            sequence_length,
        ),
    }


def main():
    tokenizer = load_tokenizer()
    save_tokenizer(tokenizer)
    tokenizer_metadata = get_tokenizer_metadata(tokenizer)
    output_path = Path(INSTRUCTION_CONFIG['output_path'])
    packers = create_packers(output_path)
    dataset = load_instruction_stream()
    total_conversations = instruction_total_conversations(dataset)
    accepted = 0
    rejected = 0
    start_time = time.monotonic()
    last_log_time = start_time

    for record in dataset:
        processed = accepted + rejected
        if (
            processed > 0
            and processed % INSTRUCTION_CONFIG['progress_check_conversations'] == 0
        ):
            current_time = time.monotonic()
            if current_time - last_log_time >= INSTRUCTION_CONFIG['progress_interval_seconds']:
                print(
                    format_instruction_progress(
                        processed,
                        total_conversations,
                        accepted,
                        rejected,
                        current_time - start_time,
                    ),
                    flush=True,
                )
                last_log_time = current_time

        messages = record.get(INSTRUCTION_CONFIG['messages_field'])
        if not isinstance(messages, list):
            rejected += 1
            continue

        token_ids, loss_mask = build_instruction_tokens(tokenizer, messages)
        if len(token_ids) > INSTRUCTION_CONFIG['maximum_conversation_tokens']:
            rejected += 1
            continue

        if sum(loss_mask) == 0:
            rejected += 1
            continue

        serialized_messages = json.dumps(
            messages,
            sort_keys=True,
            ensure_ascii=False,
        )
        document_id = stable_hash(serialized_messages)
        split = deterministic_split(
            document_id,
            INSTRUCTION_CONFIG['validation_fraction'],
        )
        packers[split].add_document(token_ids, loss_mask)
        accepted += 1

    processed = accepted + rejected
    print(
        format_instruction_progress(
            processed,
            total_conversations,
            accepted,
            rejected,
            time.monotonic() - start_time,
        ),
        flush=True,
    )

    train_statistics = packers['train'].close()
    validation_statistics = packers['validation'].close()
    statistics = [{
        'name': 'smol_smoltalk',
        'accepted_conversations': accepted,
        'rejected_conversations': rejected,
        'train': {
            key: value
            for key, value in train_statistics.items()
            if key != 'shards'
        },
        'validation': {
            key: value
            for key, value in validation_statistics.items()
            if key != 'shards'
        },
    }]

    write_manifest(
        packers['train_path'],
        tokenizer_metadata,
        INSTRUCTION_CONFIG['sequence_length'],
        statistics,
        train_statistics['shards'],
    )
    write_manifest(
        packers['validation_path'],
        tokenizer_metadata,
        INSTRUCTION_CONFIG['sequence_length'],
        statistics,
        validation_statistics['shards'],
    )

    print(f'[+] Accepted conversations: {accepted:,}')
    print(f'[+] Rejected conversations: {rejected:,}')
    print('[+] Instruction data preparation complete')


if __name__ == '__main__':
    main()
