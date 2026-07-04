from config import DOCUMENT_END_TOKEN, SPECIAL_TOKENS
from tokenizer import configure_tokenizer, get_tokenizer_metadata


class FakeTokenizer:
    def __init__(self):
        self.eos_token = '<eos>'
        self.eos_token_id = 0
        self.pad_token = None
        self.pad_token_id = None
        self.tokens = {'<eos>': 0, DOCUMENT_END_TOKEN: 1}

    def add_special_tokens(self, values):
        for token in values['additional_special_tokens']:
            if token not in self.tokens:
                self.tokens[token] = len(self.tokens)
        return len(values['additional_special_tokens'])

    def convert_tokens_to_ids(self, token):
        return self.tokens[token]

    def __len__(self):
        return len(self.tokens)


def test_configure_tokenizer():
    tokenizer = configure_tokenizer(FakeTokenizer())
    metadata = get_tokenizer_metadata(tokenizer)

    assert tokenizer.pad_token == tokenizer.eos_token
    assert metadata['vocab_size'] == 2 + len(SPECIAL_TOKENS)
    assert set(metadata['special_token_ids']) == set(SPECIAL_TOKENS)


def test_chat_mode_token_is_outside_chat_template():
    from tokenizer import format_chat_prompt

    class ChatTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            return '<chat>' + messages[0]['content']

    prompt = format_chat_prompt(
        ChatTokenizer(),
        [{'role': 'user', 'content': 'Hello'}],
        mode='reason',
    )
    assert prompt == '<|reason|>\n<chat>Hello'
