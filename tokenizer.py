import json

from config import DOCUMENT_END_TOKEN, SPECIAL_TOKENS, TOKENIZER_NAME, TOKENIZER_PATH


def configure_tokenizer(tokenizer):
    tokenizer.add_special_tokens({
        'additional_special_tokens': SPECIAL_TOKENS,
    })

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_tokenizer(local_only=False):
    from transformers import AutoTokenizer

    local_tokenizer = TOKENIZER_PATH / 'tokenizer.json'
    tokenizer_source = TOKENIZER_PATH if local_tokenizer.exists() else TOKENIZER_NAME
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=True,
        local_files_only=local_only,
    )

    return configure_tokenizer(tokenizer)


def save_tokenizer(tokenizer):
    TOKENIZER_PATH.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(TOKENIZER_PATH)

    metadata = get_tokenizer_metadata(tokenizer)
    metadata_path = TOKENIZER_PATH / 'titus_tokenizer.json'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def get_tokenizer_metadata(tokenizer):
    return {
        'name': TOKENIZER_NAME,
        'vocab_size': len(tokenizer),
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
        'document_end_token': DOCUMENT_END_TOKEN,
        'document_end_token_id': tokenizer.convert_tokens_to_ids(DOCUMENT_END_TOKEN),
        'special_tokens': SPECIAL_TOKENS,
        'special_token_ids': {
            token: tokenizer.convert_tokens_to_ids(token)
            for token in SPECIAL_TOKENS
        },
    }


def document_end_token_id(tokenizer):
    return tokenizer.convert_tokens_to_ids(DOCUMENT_END_TOKEN)


def encode_document(tokenizer, text):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_ids.append(document_end_token_id(tokenizer))
    return token_ids


def format_chat_prompt(tokenizer, messages, mode='direct'):
    mode_token = '<|reason|>' if mode == 'reason' else '<|direct|>'
    formatted_messages = []

    for message in messages:
        formatted_messages.append({
            'role': message['role'],
            'content': message['content'],
        })

    chat_text = tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return f'{mode_token}\n{chat_text}'
