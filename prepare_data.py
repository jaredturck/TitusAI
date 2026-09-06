''' Download and tokenize the training datasets. '''

import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoTokenizer

DATA_PATH = Path('data')
WEIGHTS_PATH = Path('weights')
PRETRAIN_DATA_PATH = WEIGHTS_PATH / 'data.bin'
POSTTRAIN_DATA_PATH = WEIGHTS_PATH / 'posttrain.bin'
POSTTRAIN_MASK_PATH = WEIGHTS_PATH / 'posttrain_mask.bin'
FINEWEB_REPO = 'HuggingFaceFW/fineweb_edu_100BT-shuffled'
COSMOPEDIA_REPO = 'HuggingFaceTB/smollm-corpus'
TINYSTORIES_REPO = 'maveriq/tinystoriesv2_gpt4'
POSTTRAIN_DATASET_NAME = 'open-thoughts/OpenThoughts-114k'
POSTTRAIN_DATASET_CONFIG = 'metadata'
TOKENIZER_NAME = 'gpt2'
READ_BATCH_SIZE = 65_536
TOKENIZER_BATCH_SIZE = 256
TOKENIZER_WORKERS = 12
MAX_DOCUMENT_TOKENS = 512
POSTTRAIN_CONTEXT_LENGTH = 1024

PRETRAIN_DATASETS = {
    FINEWEB_REPO: ('fineweb_edu', 299_000_000, 'data/train-{:05d}-of-00100.parquet', [42, 41, 91, 9, 65, 50, 1, 70, 15, 78, 73, 10, 55, 56, 72, 45, 48, 92, 76, 37, 30, 21, 32, 96, 80]),
    COSMOPEDIA_REPO: ('cosmopedia_v2', 115_000_000, 'cosmopedia-v2/train-{:05d}-of-00104.parquet', [84, 0, 80, 45]),
    TINYSTORIES_REPO: ('tinystories_v2', 46_000_000, 'data/train-{:05d}-of-00005.parquet', [2, 1]),
}

WORKER_TOKENIZER = None

def initialize_worker():
    ''' Load one tokenizer per worker process. '''
    global WORKER_TOKENIZER
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    WORKER_TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME, local_files_only=True)

def tokenize_batch(texts):
    ''' Tokenize a batch of documents into one packed uint16 array. '''
    encoded = WORKER_TOKENIZER(texts, add_special_tokens=False, truncation=True, max_length=MAX_DOCUMENT_TOKENS - 1, return_attention_mask=False, return_token_type_ids=False, verbose=False)['input_ids']
    tokens = np.empty(sum(len(token_ids) + 1 for token_ids in encoded), dtype=np.uint16)
    offset = 0

    for token_ids in encoded:
        length = len(token_ids)
        tokens[offset:offset + length] = token_ids
        tokens[offset + length] = WORKER_TOKENIZER.eos_token_id
        offset += length + 1

    return tokens

def prepare_pretrain():
    ''' Prepare the packed 460M-token pretraining corpus. '''
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    DATA_PATH.mkdir(exist_ok=True)
    WEIGHTS_PATH.mkdir(exist_ok=True)
    context = multiprocessing.get_context('spawn')

    with context.Pool(TOKENIZER_WORKERS, initializer=initialize_worker) as pool, PRETRAIN_DATA_PATH.open('wb') as file:
        for repo_id, (folder, target_tokens, filename_format, checkpoints) in PRETRAIN_DATASETS.items():
            written_tokens = 0
            progress = tqdm(total=target_tokens, desc=folder, unit='tok', unit_scale=True)

            for checkpoint in checkpoints:
                if written_tokens == target_tokens:
                    break

                filename = filename_format.format(checkpoint)
                local_dir = DATA_PATH / folder
                path = local_dir / filename

                if not path.exists():
                    tqdm.write(f'Downloading {folder} checkpoint {checkpoint}')
                    path = Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type='dataset', local_dir=local_dir))

                dataset = load_dataset('parquet', data_files=str(path), split='train', streaming=True)

                for batch in dataset.iter(batch_size=READ_BATCH_SIZE):
                    texts = []

                    if repo_id == FINEWEB_REPO:
                        rows = zip(batch['text'], batch['int_score'], batch['language_score'], batch['token_count'])

                        for text, int_score, language_score, token_count in rows:
                            if int_score >= 4 and language_score >= 0.95 and 100 <= token_count <= 1024:
                                text = text.replace('<|endoftext|>', ' ').strip()

                                if text:
                                    texts.append(text)
                    elif repo_id == COSMOPEDIA_REPO:
                        rows = zip(batch['text'], batch['audience'], batch['format'])

                        for text, audience, format_name in rows:
                            if audience == 'middle_school_students' and format_name.startswith('textbook'):
                                text = text.replace('<|endoftext|>', ' ').strip()

                                if text:
                                    texts.append(text)
                    else:
                        for text in batch['text']:
                            text = text.replace('<|endoftext|>', ' ').strip()

                            if text:
                                texts.append(text)

                    jobs = (texts[start:start + TOKENIZER_BATCH_SIZE] for start in range(0, len(texts), TOKENIZER_BATCH_SIZE))

                    for tokens in pool.imap(tokenize_batch, jobs, chunksize=1):
                        remaining_tokens = target_tokens - written_tokens

                        if len(tokens) > remaining_tokens:
                            tokens = tokens[:remaining_tokens].copy()
                            tokens[-1] = tokenizer.eos_token_id

                        tokens.tofile(file)
                        written_tokens += len(tokens)
                        progress.update(len(tokens))

                        if written_tokens == target_tokens:
                            break

                    if written_tokens == target_tokens:
                        break

            progress.close()

            if written_tokens != target_tokens:
                raise RuntimeError(f'{folder} produced {written_tokens:,} / {target_tokens:,} tokens')

    print(f'Saved 460,000,000 tokens to {PRETRAIN_DATA_PATH}')

def write_reasoning_batch(tokenizer, data_file, mask_file, batch):
    ''' Write fixed reasoning samples and assistant loss masks. '''
    prompts = [f'User:\n{problem}\nAssistant:\n<think>\n' for problem in batch['problem']]
    responses = [f'{reasoning}\n</think>\n{solution}' for reasoning, solution in zip(batch['deepseek_reasoning'], batch['deepseek_solution'])]
    prompt_ids = tokenizer(prompts, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False, verbose=False)['input_ids']
    response_ids = tokenizer(responses, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False, verbose=False)['input_ids']
    sequence_length = POSTTRAIN_CONTEXT_LENGTH + 1
    tokens = np.full((len(prompts), sequence_length), tokenizer.eos_token_id, dtype=np.uint16)
    masks = np.zeros((len(prompts), sequence_length), dtype=np.uint8)
    sample_count = 0

    for prompt, response in zip(prompt_ids, response_ids):
        response = response + [tokenizer.eos_token_id]
        length = len(prompt) + len(response)

        if length > sequence_length:
            continue

        tokens[sample_count, :len(prompt)] = prompt
        tokens[sample_count, len(prompt):length] = response
        masks[sample_count, len(prompt):length] = 1
        sample_count += 1

    tokens[:sample_count].tofile(data_file)
    masks[:sample_count].tofile(mask_file)
    return sample_count

def prepare_posttrain():
    ''' Prepare masked OpenThoughts reasoning samples. '''
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    dataset = load_dataset(POSTTRAIN_DATASET_NAME, POSTTRAIN_DATASET_CONFIG, split='train')
    sample_count = 0
    WEIGHTS_PATH.mkdir(exist_ok=True)

    with POSTTRAIN_DATA_PATH.open('wb') as data_file, POSTTRAIN_MASK_PATH.open('wb') as mask_file, tqdm(total=len(dataset), desc='Preparing', unit='samples') as progress:
        for batch in dataset.iter(batch_size=TOKENIZER_BATCH_SIZE):
            sample_count += write_reasoning_batch(tokenizer, data_file, mask_file, batch)
            progress.update(len(batch['problem']))

    print(f'Saved {sample_count:,} reasoning samples to {POSTTRAIN_DATA_PATH}')

if __name__ == '__main__':
    stages = ('pretrain', 'posttrain') if sys.argv[1] == 'all' else (sys.argv[1],)

    for stage in stages:
        assert stage in ('pretrain', 'posttrain')

        if stage == 'pretrain':
            prepare_pretrain()
        else:
            prepare_posttrain()
