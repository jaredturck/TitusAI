from prepare_instructions import build_instruction_tokens


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
