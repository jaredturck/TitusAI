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
OASST_REPO = 'OpenAssistant/oasst1'
SMOLTALK_REPO = 'HuggingFaceTB/smoltalk'
SMOLTALK_CONFIG = 'everyday-conversations'
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

CODE_MARKERS = ('```', 'python', 'javascript', 'function', 'write code', 'programming', 'algorithm', 'class ')
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

def build_oasst_conversations(dataset):
    ''' Build strict English human conversation paths ending in top-ranked responses. '''
    by_id = {row['message_id']: row for row in dataset}
    conversations = []

    for row in dataset:
        if row['role'] != 'assistant' or row['lang'] != 'en':
            continue
        if row['deleted'] or not row['review_result'] or row['synthetic'] or row['rank'] != 0:
            continue

        messages = []
        current = row

        while current is not None:
            if current['lang'] != 'en' or current['deleted'] or not current['review_result'] or current['synthetic']:
                messages = []
                break

            role = 'user' if current['role'] == 'prompter' else 'assistant'
            messages.append({'role': role, 'content': current['text']})
            current = by_id.get(current['parent_id']) if current['parent_id'] else None

        messages.reverse()

        if len(messages) >= 2 and messages[-1]['role'] == 'assistant':
            conversations.append(messages)

    return conversations

def is_code_heavy(messages):
    ''' Reject conversations whose user prompts are clearly code-focused. '''
    user_text = '\n'.join(message['content'] or '' for message in messages if message['role'] == 'user').lower()
    return any(marker in user_text for marker in CODE_MARKERS)

def encode_conversation(tokenizer, messages, last_assistant_only=False):
    ''' Encode one conversation and mark assistant response tokens for loss. '''
    token_ids = []
    loss_mask = []
    last_assistant = max(index for index, message in enumerate(messages) if message['role'] == 'assistant')

    for index, message in enumerate(messages):
        role = message['role']
        content = (message['content'] or '').replace('<|endoftext|>', ' ').strip()

        if role not in ('user', 'assistant') or not content:
            return None

        prefix = ('' if not token_ids else '\n') + ('User:\n' if role == 'user' else 'Assistant:\n')
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        token_ids.extend(prefix_ids)
        loss_mask.extend([0] * len(prefix_ids))
        token_ids.extend(content_ids)

        train_response = role == 'assistant' and (not last_assistant_only or index == last_assistant)
        loss_mask.extend([1 if train_response else 0] * len(content_ids))

        if role == 'assistant':
            token_ids.append(tokenizer.eos_token_id)
            loss_mask.append(1 if train_response else 0)

    if len(token_ids) > POSTTRAIN_CONTEXT_LENGTH + 1:
        return None

    return token_ids, loss_mask

def write_conversation(tokenizer, data_file, mask_file, messages, last_assistant_only=False):
    ''' Write one fixed-length conversation and assistant loss mask. '''
    encoded = encode_conversation(tokenizer, messages, last_assistant_only)

    if encoded is None:
        return False

    token_ids, loss_mask = encoded
    sequence_length = POSTTRAIN_CONTEXT_LENGTH + 1
    tokens = np.full(sequence_length, tokenizer.eos_token_id, dtype=np.uint16)
    masks = np.zeros(sequence_length, dtype=np.uint8)
    tokens[:len(token_ids)] = token_ids
    masks[:len(loss_mask)] = loss_mask
    tokens.tofile(data_file)
    masks.tofile(mask_file)
    return True

def prepare_posttrain():
    ''' Prepare filtered conversational assistant fine-tuning samples. '''
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f'Loading {OASST_REPO}')
    oasst = load_dataset(OASST_REPO, split='train')
    oasst_conversations = build_oasst_conversations(oasst)
    print(f'Loading {SMOLTALK_REPO} ({SMOLTALK_CONFIG})')
    everyday = load_dataset(SMOLTALK_REPO, SMOLTALK_CONFIG, split='train')
    WEIGHTS_PATH.mkdir(exist_ok=True)

    oasst_written = 0
    oasst_code_skipped = 0
    oasst_length_skipped = 0
    everyday_written = 0
    everyday_skipped = 0

    with POSTTRAIN_DATA_PATH.open('wb') as data_file, POSTTRAIN_MASK_PATH.open('wb') as mask_file:
        for messages in tqdm(oasst_conversations, desc='OASST1', unit='samples'):
            if is_code_heavy(messages):
                oasst_code_skipped += 1
                continue

            if write_conversation(tokenizer, data_file, mask_file, messages, last_assistant_only=True):
                oasst_written += 1
            else:
                oasst_length_skipped += 1

        for row in tqdm(everyday, desc='Everyday conversations', unit='samples'):
            if write_conversation(tokenizer, data_file, mask_file, row['messages']):
                everyday_written += 1
            else:
                everyday_skipped += 1

    total = oasst_written + everyday_written
    print(f'Saved {total:,} post-training samples to {POSTTRAIN_DATA_PATH}')
    print(f'OASST1: {oasst_written:,} kept, {oasst_code_skipped:,} code-heavy skipped, {oasst_length_skipped:,} over-length/invalid skipped')
    print(f'Everyday conversations: {everyday_written:,} kept, {everyday_skipped:,} over-length/invalid skipped')

if __name__ == '__main__':
    stages = ('pretrain', 'posttrain') if sys.argv[1] == 'all' else (sys.argv[1],)

    for stage in stages:
        assert stage in ('pretrain', 'posttrain')

        if stage == 'pretrain':
            prepare_pretrain()
        else:
            prepare_posttrain()
