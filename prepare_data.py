''' Download and tokenize the training dataset. '''

from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

WEIGHTS_PATH = Path('weights')
DATA_PATH = WEIGHTS_PATH / 'data.pt'
DATASET_NAME = 'Salesforce/wikitext'
DATASET_CONFIG = 'wikitext-103-raw-v1'
TOKENIZER_NAME = 'gpt2'

def tokenize(batch):
    ''' Tokenize a batch of training text. '''
    return tokenizer(batch['text'], add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)

WEIGHTS_PATH.mkdir(exist_ok=True)

dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split='train')
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names, desc='Tokenizing')

tokens = np.concatenate(tokenized['input_ids'], dtype=np.int32)
tokens = torch.from_numpy(tokens)

torch.save(tokens, DATA_PATH)
print(f'Saved {len(tokens):,} tokens to {DATA_PATH}')
