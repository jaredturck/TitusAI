''' Download and tokenize the training dataset. '''

from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

WEIGHTS_PATH = Path('weights')
DATA_PATH = WEIGHTS_PATH / 'data.bin'
DATASET_NAME = 'Salesforce/wikitext'
DATASET_CONFIG = 'wikitext-103-raw-v1'
TOKENIZER_NAME = 'gpt2'
READ_BATCH_SIZE = 10000
TOKENIZER_BATCH_SIZE = 512

def is_article_title(text):
    ''' Check whether a WikiText row starts a new article. '''
    text = text.strip()
    return text.startswith('= ') and text.endswith(' =') and not text.startswith('= =')

def write_documents(file, documents, tokenizer):
    ''' Tokenize documents and append them to the packed token stream. '''
    encoded = tokenizer(documents, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False, verbose=False)['input_ids']
    token_count = sum(len(token_ids) + 1 for token_ids in encoded)
    tokens = np.empty(token_count, dtype=np.uint16)
    offset = 0

    for token_ids in encoded:
        length = len(token_ids)
        tokens[offset:offset + length] = token_ids
        tokens[offset + length] = tokenizer.eos_token_id
        offset += length + 1

    tokens.tofile(file)
    return token_count

WEIGHTS_PATH.mkdir(exist_ok=True)

dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split='train')
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
documents = []
document = []
token_count = 0
article_count = 0

with DATA_PATH.open('wb') as file, tqdm(total=len(dataset), desc='Preparing', unit='lines') as progress:
    for batch in dataset.iter(batch_size=READ_BATCH_SIZE):
        for text in batch['text']:
            if is_article_title(text):
                if document:
                    documents.append('\n'.join(document) + '\n')
                    article_count += 1
                document = [text]
            elif document:
                document.append(text)

            if len(documents) >= TOKENIZER_BATCH_SIZE:
                token_count += write_documents(file, documents, tokenizer)
                documents.clear()

        progress.update(len(batch['text']))

    if document:
        documents.append('\n'.join(document) + '\n')
        article_count += 1

    if documents:
        token_count += write_documents(file, documents, tokenizer)

print(f'Saved {token_count:,} tokens from {article_count:,} articles to {DATA_PATH}')
