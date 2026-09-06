''' Build the finalized TitusAI pre-training corpus. '''

import multiprocessing
import os
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT_PATH = Path(__file__).parent
DATA_PATH = ROOT_PATH / 'data'
OUTPUT_PATH = ROOT_PATH / 'pretrain_corpus' / 'pretrain.bin'
TOKENIZER_NAME = 'gpt2'
TOTAL_TOKENS = 460_000_000
MAX_DOCUMENT_TOKENS = 512
PARQUET_BATCH_ROWS = 65_536
TOKENIZE_BATCH_SIZE = 256
WORKER_COUNT = max(1, (os.cpu_count() or 2) // 2)
SEED = 42
EOS_TOKEN_ID = 50256

SOURCES = (
    {
        'name': 'fineweb_edu',
        'label': 'FineWeb-Edu',
        'dataset': 'HuggingFaceFW/fineweb_edu_100BT-shuffled',
        'path_prefix': 'data/',
        'target_tokens': 299_000_000,
        'columns': ('text', 'int_score', 'language_score', 'token_count'),
    },
    {
        'name': 'cosmopedia_v2',
        'label': 'Cosmopedia v2',
        'dataset': 'HuggingFaceTB/smollm-corpus',
        'path_prefix': 'cosmopedia-v2/',
        'target_tokens': 115_000_000,
        'columns': ('text', 'audience', 'format'),
    },
    {
        'name': 'tinystories_v2',
        'label': 'TinyStories V2',
        'dataset': 'maveriq/tinystoriesv2_gpt4',
        'path_prefix': 'data/train-',
        'target_tokens': 46_000_000,
        'columns': ('text',),
    },
)

WORKER_TOKENIZER = None


def initialize_worker():
    global WORKER_TOKENIZER
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    WORKER_TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME, local_files_only=True)


def tokenize_text_batch(texts):
    encoded = WORKER_TOKENIZER(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_DOCUMENT_TOKENS - 1,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )['input_ids']
    documents = []

    for token_ids in encoded:
        documents.append(np.asarray(token_ids + [EOS_TOKEN_ID], dtype=np.uint16))

    return documents


def get_source_files(source):
    files = [
        filename for filename in list_repo_files(source['dataset'], repo_type='dataset')
        if filename.startswith(source['path_prefix']) and filename.endswith('.parquet')
    ]
    random.Random(SEED + SOURCES.index(source)).shuffle(files)
    return files


def get_local_shard(source, filename, checkpoint):
    local_path = DATA_PATH / source['name'] / filename

    if local_path.exists():
        return local_path

    tqdm.write(f'Downloading {source["label"]} checkpoint {checkpoint}...')
    return Path(hf_hub_download(
        repo_id=source['dataset'],
        filename=filename,
        repo_type='dataset',
        local_dir=DATA_PATH / source['name'],
    ))


def filter_batch(source, batch):
    table = pa.Table.from_batches([batch])

    if source['name'] == 'fineweb_edu':
        mask = pc.and_kleene(
            pc.and_kleene(
                pc.greater_equal(table['int_score'], 4),
                pc.greater_equal(table['language_score'], 0.95),
            ),
            pc.and_kleene(
                pc.greater_equal(table['token_count'], 100),
                pc.less_equal(table['token_count'], 1024),
            ),
        )
        table = table.filter(mask)
    elif source['name'] == 'cosmopedia_v2':
        mask = pc.and_kleene(
            pc.equal(table['audience'], 'middle_school_students'),
            pc.starts_with(table['format'], 'textbook'),
        )
        table = table.filter(mask)

    texts = table['text'].to_pylist()
    texts = [text.replace('<|endoftext|>', ' ').strip() for text in texts if text]
    return [text for text in texts if text]


def tokenize_jobs(texts):
    for start in range(0, len(texts), TOKENIZE_BATCH_SIZE):
        yield texts[start:start + TOKENIZE_BATCH_SIZE]


def write_documents(file, documents, remaining_tokens):
    written = 0

    for tokens in documents:
        if remaining_tokens == 0:
            break

        if len(tokens) <= remaining_tokens:
            tokens.tofile(file)
            written += len(tokens)
            remaining_tokens -= len(tokens)
            continue

        final_document = tokens[:remaining_tokens].copy()
        final_document[-1] = EOS_TOKEN_ID
        final_document.tofile(file)
        written += remaining_tokens
        remaining_tokens = 0

    return written


def build_source(source, file, pool):
    target_tokens = source['target_tokens']
    source_tokens = 0
    checkpoint_count = 0
    progress = tqdm(total=target_tokens, desc=source['label'], unit='tok', unit_scale=True)

    for shard_index, filename in enumerate(get_source_files(source)):
        if source_tokens == target_tokens:
            break

        checkpoint_count += 1
        progress.set_description(f'{source["label"]} checkpoint {checkpoint_count}')
        shard_path = get_local_shard(source, filename, checkpoint_count)
        parquet = pq.ParquetFile(shard_path)
        random_generator = random.Random(SEED + shard_index + 10_000 * (SOURCES.index(source) + 1))

        for batch in parquet.iter_batches(
            batch_size=PARQUET_BATCH_ROWS,
            columns=list(source['columns']),
            use_threads=True,
        ):
            texts = filter_batch(source, batch)
            random_generator.shuffle(texts)

            for documents in pool.imap(tokenize_text_batch, tokenize_jobs(texts), chunksize=1):
                remaining = target_tokens - source_tokens
                written = write_documents(file, documents, remaining)
                source_tokens += written
                progress.update(written)

                if source_tokens == target_tokens:
                    break

            if source_tokens == target_tokens:
                break

    progress.close()
    print(f'{source["label"]} complete: {checkpoint_count} checkpoints processed')

    if source_tokens != target_tokens:
        raise RuntimeError(f'{source["label"]} produced {source_tokens:,} / {target_tokens:,} tokens.')


def main():
    if sum(source['target_tokens'] for source in SOURCES) != TOTAL_TOKENS:
        raise RuntimeError('Source token quotas do not add up to TOTAL_TOKENS.')

    DATA_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    context = multiprocessing.get_context('spawn')

    print(f'CPU tokenizer workers: {WORKER_COUNT}')
    print(f'Final corpus: {TOTAL_TOKENS:,} uint16 tokens ({TOTAL_TOKENS * 2 / 1024 ** 3:.2f} GiB)')

    with context.Pool(WORKER_COUNT, initializer=initialize_worker) as pool:
        with OUTPUT_PATH.open('wb') as file:
            for source in SOURCES:
                build_source(source, file, pool)

    print(f'Wrote {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
