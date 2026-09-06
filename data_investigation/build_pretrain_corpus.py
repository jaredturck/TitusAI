''' Build the TitusAI pre-training corpus. '''

from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

FINEWEB_REPO = 'HuggingFaceFW/fineweb_edu_100BT-shuffled'
COSMOPEDIA_REPO = 'HuggingFaceTB/smollm-corpus'
TINYSTORIES_REPO = 'maveriq/tinystoriesv2_gpt4'
MAX_DOCUMENT_TOKENS = 512
EOS_TOKEN_ID = 50256

DATASETS = {
    FINEWEB_REPO: (299_000_000, 'data/train-{:05d}-of-00100.parquet', [42,41,91,9,65,50,1,70,15,78,73,10,55,56,72,45,48,92,76,37,30,21,32,96,80]),
    COSMOPEDIA_REPO: (115_000_000, 'cosmopedia-v2/train-{:05d}-of-00104.parquet', [84, 0, 80, 45]),
    TINYSTORIES_REPO: (46_000_000, 'data/train-{:05d}-of-00005.parquet', [2, 1]),
}

Path('pretrain_corpus').mkdir(exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained('gpt2')

with open('pretrain_corpus/pretrain.bin', 'wb') as output:
    for repo_id, (target_tokens, filename_format, checkpoints) in DATASETS.items():
        written_tokens = 0

        for checkpoint in checkpoints:
            filename = filename_format.format(checkpoint)
            path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type='dataset')
            dataset = load_dataset('parquet', data_files=path, split='train', keep_in_memory=True)

            for row in dataset:
                if repo_id == FINEWEB_REPO:
                    if row['int_score'] < 4 or row['language_score'] < 0.95 or row['token_count'] < 100 or row['token_count'] > 1024:
                        continue
                elif repo_id == COSMOPEDIA_REPO:
                    if row['audience'] != 'middle_school_students' or not row['format'].startswith('textbook'):
                        continue

                text = row['text'].replace('<|endoftext|>', ' ').strip()

                if not text:
                    continue

                remaining_tokens = target_tokens - written_tokens
                tokens = tokenizer.encode(text, add_special_tokens=False)[:MAX_DOCUMENT_TOKENS - 1]
                tokens = tokens[:remaining_tokens - 1]
                tokens.append(EOS_TOKEN_ID)
                output.write(np.asarray(tokens, dtype=np.uint16).tobytes())
                written_tokens += len(tokens)

                if written_tokens == target_tokens:
                    break

            if written_tokens == target_tokens:
                break
