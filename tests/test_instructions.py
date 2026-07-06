import sys
from types import SimpleNamespace

from config import INSTRUCTION_CONFIG, INSTRUCTION_SOURCES
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


def test_extracts_daily_dialogue():
    source = {
        'messages_field': 'dialog',
        'message_text_field': None,
    }
    messages = extract_conversation_messages({
        'dialog': ['How are you?', 'Pretty good.'],
    }, source)

    assert messages == ['How are you?', 'Pretty good.']


def test_instruction_stream_uses_huggingface_datasets(monkeypatch):
    captured = {}
    dataset = object()

    def load_dataset(**arguments):
        captured.update(arguments)
        return dataset

    monkeypatch.setitem(
        sys.modules,
        'datasets',
        SimpleNamespace(load_dataset=load_dataset),
    )
    source = {
        'dataset': 'example/conversations',
        'config': None,
        'split': 'train',
    }

    result = load_instruction_stream(source, shuffle=False)

    assert result is dataset
    assert captured == {
        'path': 'example/conversations',
        'split': 'train',
        'streaming': True,
    }


def test_instruction_sources_use_soda_and_daily_dialogue():
    targets = {
        source['name']: source['target_tokens']
        for source in INSTRUCTION_SOURCES
    }

    assert targets == {
        'soda': 45_000_000,
        'daily_dialog': 5_000_000,
    }
    assert sum(targets.values()) == INSTRUCTION_CONFIG['max_total_tokens']


def test_instruction_progress_includes_token_target_speed_and_eta():
    progress = format_instruction_progress(
        'soda',
        400_000,
        390_000,
        10_000,
        36_000_000,
        45_000_000,
        1000,
    )

    assert 'soda: 36,000,000 / 45,000,000 tokens (80.00%)' in progress
    assert 'processed=400,000 accepted=390,000 rejected=10,000' in progress
    assert '400.0 conversations/s' in progress
    assert 'ETA=' in progress


def test_conversation_packers_do_not_store_loss_masks(tmp_path):
    packers = create_source_packers(tmp_path, 'conversation')
    packers['train'].add_document([1] * 2049)
    train_statistics = packers['train'].close()
    packers['validation'].close()

    assert train_statistics['shards'][0]['loss_mask'] is None
    assert not list((tmp_path / 'train').glob('*.loss.bin'))
