from prepare_instructions import (
    build_instruction_tokens,
    format_instruction_progress,
)


class InstructionTokenizer:
    def __init__(self):
        self.eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(character) % 251 + 1 for character in text]

    def convert_tokens_to_ids(self, token):
        if token == '<|endoftext|>':
            return 0
        return 1


def test_instruction_without_assistant_has_no_loss():
    tokenizer = InstructionTokenizer()
    _, loss_mask = build_instruction_tokens(tokenizer, [
        {
            'role': 'user',
            'content': 'Hello',
        },
    ])
    assert sum(loss_mask) == 0


def test_instruction_masks_only_assistant_tokens():
    tokenizer = InstructionTokenizer()
    _, loss_mask = build_instruction_tokens(tokenizer, [
        {
            'role': 'user',
            'content': 'Question',
        },
        {
            'role': 'assistant',
            'content': 'Answer',
        },
    ])
    assert sum(loss_mask) > 0
    assert loss_mask[-1] == 1


def test_instruction_progress_includes_total_speed_and_eta():
    progress = format_instruction_progress(
        400_000,
        500_000,
        390_000,
        10_000,
        1000,
    )

    assert '400,000 / 500,000 conversations (80.00%)' in progress
    assert 'accepted=390,000 rejected=10,000' in progress
    assert '400.0 conversations/s' in progress
    assert 'ETA=4m 10s' in progress


def test_instruction_progress_works_without_dataset_total():
    progress = format_instruction_progress(10_000, None, 9_000, 1_000, 100)

    assert 'Processed 10,000 conversations' in progress
    assert '%' not in progress
    assert 'ETA=' not in progress
