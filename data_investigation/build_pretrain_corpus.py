''' Build the TitusAI pre-training corpus. '''

from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

FINEWEB_REPO = 'HuggingFaceFW/fineweb_edu_100BT-shuffled'
COSMOPEDIA_REPO = 'HuggingFaceTB/smollm-corpus'
TINYSTORIES_REPO = 'maveriq/tinystoriesv2_gpt4'
FINEWEB_END = 299_000_000
COSMOPEDIA_END = 414_000_000
TINYSTORIES_END = 460_000_000
MAX_DOCUMENT_TOKENS = 512
EOS_TOKEN_ID = 50256

FILES = [
    (FINEWEB_REPO, 'data/train-00042-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00041-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00091-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00009-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00065-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00050-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00001-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00070-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00015-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00078-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00073-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00010-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00055-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00056-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00072-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00045-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00048-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00092-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00076-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00037-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00030-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00021-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00032-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00096-of-00100.parquet'),
    (FINEWEB_REPO, 'data/train-00080-of-00100.parquet'),
    (COSMOPEDIA_REPO, 'cosmopedia-v2/train-00084-of-00104.parquet'),
    (COSMOPEDIA_REPO, 'cosmopedia-v2/train-00000-of-00104.parquet'),
    (COSMOPEDIA_REPO, 'cosmopedia-v2/train-00080-of-00104.parquet'),
    (COSMOPEDIA_REPO, 'cosmopedia-v2/train-00045-of-00104.parquet'),
    (TINYSTORIES_REPO, 'data/train-00002-of-00005.parquet'),
    (TINYSTORIES_REPO, 'data/train-00001-of-00005.parquet'),
]

Path('pretrain_corpus').mkdir(exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained('gpt2')
written_tokens = 0

with open('pretrain_corpus/pretrain.bin', 'wb') as output:
    for repo_id, filename in FILES:
        if repo_id == FINEWEB_REPO:
            target_tokens = FINEWEB_END
        elif repo_id == COSMOPEDIA_REPO:
            target_tokens = COSMOPEDIA_END
        else:
            target_tokens = TINYSTORIES_END

        if written_tokens >= target_tokens:
            continue

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
