import torch

from config import DATA_SOURCES, MODEL_CONFIG, SPECIAL_TOKENS
from data_utils import extract_first
from model import TitusModel
from prepare_data import load_source_stream
from process_utils import hard_exit_after_success
from tokenizer import document_end_token_id, get_tokenizer_metadata, load_tokenizer, save_tokenizer


def check_tokenizer():
    tokenizer = load_tokenizer()
    save_tokenizer(tokenizer)
    metadata = get_tokenizer_metadata(tokenizer)

    for token in SPECIAL_TOKENS:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        assert len(token_ids) == 1, f'{token} must encode to one token'

    print(f'[+] Tokenizer vocabulary: {len(tokenizer):,}')
    print(f'[+] Chat EOS token: {tokenizer.eos_token} ({tokenizer.eos_token_id})')
    print(f'[+] Document end token ID: {document_end_token_id(tokenizer)}')
    return tokenizer, metadata


def check_model(metadata):
    model_config = dict(MODEL_CONFIG)
    model_config['vocab_size'] = metadata['vocab_size']

    with torch.device('meta'):
        model = TitusModel(model_config)

    print(f'[+] Model parameters: {model.parameter_count():,}')


def check_datasets():
    for source in DATA_SOURCES:
        print(f'[+] Checking {source["name"]}...')
        dataset = load_source_stream(source, shuffle=False)
        iterator = iter(dataset)

        try:
            record = next(iterator)
        finally:
            close = getattr(iterator, 'close', None)
            if close is not None:
                close()

        text = extract_first(record, source['text_fields'])
        assert isinstance(text, str)
        assert text.strip()
        print(
            f'[+] {source["name"]}: '
            f'{len(text):,} characters, fields={sorted(record.keys())}'
        )


def main():
    _, metadata = check_tokenizer()
    check_model(metadata)
    check_datasets()
    print('[+] Setup check complete')


def run():
    main()
    hard_exit_after_success()


if __name__ == '__main__':
    run()
