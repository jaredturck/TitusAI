''' Download and tokenize the training dataset. '''

from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

WEIGHTS_PATH = Path('weights')
DATA_PATH = WEIGHTS_PATH / 'data.pt'
DATASET_NAME = 'Salesforce/wikitext'
DATASET_CONFIG = 'wikitext-103-raw-v1'
TOKENIZER_NAME = 'gpt2'

WEIGHTS_PATH.mkdir(exist_ok=True)

dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split='train')
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
text = '\n'.join(dataset['text'])
tokens = tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids'][0].to(torch.int32)

torch.save(tokens, DATA_PATH)
print(f'Saved {len(tokens):,} tokens to {DATA_PATH}')
