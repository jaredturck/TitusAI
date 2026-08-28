''' Download and tokenize the training dataset. '''

from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

WEIGHTS_PATH = Path('weights')
DATA_PATH = WEIGHTS_PATH / 'data.pt'
DATASET_NAME = 'Salesforce/wikitext'
DATASET_CONFIG = 'wikitext-103-raw-v1'
TOKENIZER_NAME = 'gpt2'
CHUNK_SIZE = 1000

WEIGHTS_PATH.mkdir(exist_ok=True)

dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split='train')
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
chunks = []

for start in tqdm(range(0, len(dataset), CHUNK_SIZE), desc='Tokenizing', unit='chunks'):
    texts = dataset[start:start + CHUNK_SIZE]['text']
    text = '\n\n'.join(text for text in texts if text.strip()) + '\n\n'

    if not text.strip():
        continue

    tokens = tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
    chunks.append(tokens.to(torch.int32))

tokens = torch.cat(chunks)
torch.save(tokens, DATA_PATH)
print(f'Saved {len(tokens):,} tokens to {DATA_PATH}')
