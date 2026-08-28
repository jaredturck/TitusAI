from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


Path('weights').mkdir(exist_ok=True)

dataset = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
chunks = []

for start in range(0, len(dataset), 1000):
    texts = dataset[start:start + 1000]['text']
    text = '\n\n'.join(text for text in texts if text.strip()) + '\n\n'

    if not text.strip():
        continue

    tokens = tokenizer(text, add_special_tokens=False, return_tensors='pt')['input_ids'][0]
    chunks.append(tokens.to(torch.int32))

tokens = torch.cat(chunks)
torch.save(tokens, 'weights/data.pt')
print(f'Saved {len(tokens):,} tokens to weights/data.pt')
