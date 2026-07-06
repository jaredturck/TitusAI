import io
import json

import huggingface_hub

from prepare_instructions import (
    build_instruction_tokens,
    create_source_packers,
    extract_conversation_messages,
    format_instruction_progress,
    load_instruction_stream,
)


class InstructionTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) % 251 + 1 for character in text]

    def convert_tokens_to_ids(self, token):
        if token == '<|endoftext|>':
            return 0
        return 1


def test_conversation_tokens_use_newlines_without_role_prefixes():
    tokenizer = InstructionTokenizer()
    token_ids = build_instruction_tokens(tokenizer, [
        'Hey you',
        'Morning, sleepyhead',
    ])
    expected = tokenizer.encode(
        'Hey you\nMorning, sleepyhead',
        add_special_tokens=False,
    ) + [0]

    assert token_ids == expected


def test_extracts_soda_dialogue():
    source = {
        'messages_field': 'dialogue',
        'message_text_field': None,
    }
    messages = extract_conversation_messages({
        'dialogue': ['Hello', 'Hi there'],
    }, source)

    assert messages == ['Hello', 'Hi there']


def test_extracts_nested_topical_chat_dialogue():
    source = {
        'messages_field': 'content',
        'message_text_field': 'message',
    }
    messages = extract_conversation_messages({
        'conversation-id': {
            'content': [
                {'agent': 'agent_1', 'message': 'Hello\nthere'},
                {'agent': 'agent_2', 'message': 'Hi'},
            ],
        },
    }, source)

    assert messages == ['Hello there', 'Hi']


def test_extracts_daily_dialogue():
    source = {
        'messages_field': 'dialog',
        'message_text_field': None,
    }
    messages = extract_conversation_messages({
        'dialog': ['How are you?', 'Pretty good.'],
    }, source)

    assert messages == ['How are you?', 'Pretty good.']


def test_instruction_progress_includes_token_target_speed_and_eta():
    progress = format_instruction_progress(
        'soda',
        400_000,
        390_000,
        10_000,
        32_000_000,
        40_000_000,
        1000,
    )

    assert 'soda: 32,000,000 / 40,000,000 tokens (80.00%)' in progress
    assert 'processed=400,000 accepted=390,000 rejected=10,000' in progress
    assert '400.0 conversations/s' in progress
    assert 'ETA=' in progress


class ConversationFileSystem:
    def open(self, filename, mode, block_size):
        assert filename == 'datasets/example/topical/train.jsonl'
        assert mode == 'rb'
        assert block_size == 8 * 1024 * 1024
        record = {
            'conversation-1': {
                'content': [
                    {'message': 'Hello'},
                    {'message': 'Hi there'},
                ],
            },
        }
        return io.BytesIO(json.dumps(record).encode('utf-8') + b'\n')


def test_jsonl_conversation_stream_reads_topical_chat(monkeypatch):
    monkeypatch.setattr(huggingface_hub, 'HfFileSystem', ConversationFileSystem)
    source = {
        'dataset': 'example/topical',
        'loader': 'jsonl',
        'data_files': ['train.jsonl'],
        'messages_field': 'content',
        'message_text_field': 'message',
    }

    records = list(load_instruction_stream(source, shuffle=False))
    messages = extract_conversation_messages(records[0], source)

    assert messages == ['Hello', 'Hi there']


def test_conversation_packers_do_not_store_loss_masks(tmp_path):
    packers = create_source_packers(tmp_path, 'conversation')
    packers['train'].add_document([1] * 2049)
    train_statistics = packers['train'].close()
    packers['validation'].close()

    assert train_statistics['shards'][0]['loss_mask'] is None
    assert not list((tmp_path / 'train').glob('*.loss.bin'))
