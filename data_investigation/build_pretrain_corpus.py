''' Build the finalized TitusAI pre-training corpus. '''

import hashlib
import json
import multiprocessing
import os
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT_PATH = Path(__file__).parent
DATA_PATH = ROOT_PATH / 'data'
BUILD_PATH = ROOT_PATH / 'pretrain_corpus'
PLAN_PATH = BUILD_PATH / 'plan.json'
OUTPUT_PATH = BUILD_PATH / 'pretrain.bin'
MANIFEST_PATH = BUILD_PATH / 'pretrain.json'
TOKENIZER_NAME = 'gpt2'
TOTAL_TOKENS = 460_000_000
MAX_DOCUMENT_TOKENS = 512
PARQUET_BATCH_ROWS = 65_536
TOKENIZE_BATCH_SIZE = 256
WORKER_COUNT = max(1, (os.cpu_count() or 2) // 2)
SEED = 42
EOS_TOKEN_ID = 50256
UINT16_BYTES = 2
COPY_BUFFER_SIZE = 16 * 1024 * 1024

SOURCES = (
    {
        'name': 'fineweb_edu',
        'label': 'FineWeb-Edu',
        'dataset': 'HuggingFaceFW/fineweb_edu_100BT-shuffled',
        'path_prefix': 'data/',
        'target_tokens': 299_000_000,
        'columns': ('text', 'int_score', 'language_score', 'token_count'),
        'filter': 'int_score >= 4, language_score >= 0.95, 100 <= token_count <= 1024',
    },
    {
        'name': 'cosmopedia_v2',
        'label': 'Cosmopedia v2 middle-school textbooks',
        'dataset': 'HuggingFaceTB/smollm-corpus',
        'path_prefix': 'cosmopedia-v2/',
        'target_tokens': 115_000_000,
        'columns': ('text', 'audience', 'format'),
        'filter': 'audience == middle_school_students, format starts with textbook',
    },
    {
        'name': 'tinystories_v2',
        'label': 'TinyStories V2 GPT-4',
        'dataset': 'maveriq/tinystoriesv2_gpt4',
        'path_prefix': 'data/train-',
        'target_tokens': 46_000_000,
        'columns': ('text',),
        'filter': 'non-empty text',
    },
)

WORKER_TOKENIZER = None


def atomic_write_json(path, data):
    ''' Write JSON atomically. '''
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    temporary_path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary_path.replace(path)


def initialize_worker():
    ''' Load one tokenizer per worker process. '''
    global WORKER_TOKENIZER
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    WORKER_TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME, local_files_only=True)
    WORKER_TOKENIZER.model_max_length = 1_000_000_000


def tokenize_text_batch(texts):
    ''' Tokenize one document batch and return flat uint16 tokens plus lengths. '''
    encoded = WORKER_TOKENIZER(
        texts,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_DOCUMENT_TOKENS - 1,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )['input_ids']
    lengths = np.empty(len(encoded), dtype=np.uint16)
    total_tokens = sum(len(token_ids) + 1 for token_ids in encoded)
    tokens = np.empty(total_tokens, dtype=np.uint16)
    offset = 0

    for index, token_ids in enumerate(encoded):
        length = len(token_ids) + 1
        lengths[index] = length
        tokens[offset:offset + length - 1] = token_ids
        tokens[offset + length - 1] = EOS_TOKEN_ID
        offset += length

    return tokens, lengths


def source_config_for_plan(source):
    ''' Return stable source configuration recorded in the build plan. '''
    return {
        'name': source['name'],
        'label': source['label'],
        'dataset': source['dataset'],
        'path_prefix': source['path_prefix'],
        'target_tokens': source['target_tokens'],
        'filter': source['filter'],
    }


def create_plan():
    ''' Freeze dataset revisions and deterministic shard order. '''
    api = HfApi()
    plan = {
        'version': 1,
        'seed': SEED,
        'tokenizer': TOKENIZER_NAME,
        'total_tokens': TOTAL_TOKENS,
        'max_document_tokens': MAX_DOCUMENT_TOKENS,
        'sources': {},
    }

    for source_index, source in enumerate(SOURCES):
        print(f'Resolving {source["label"]}...')
        revision = api.dataset_info(source['dataset']).sha
        files = [
            filename for filename in list_repo_files(
                source['dataset'],
                repo_type='dataset',
                revision=revision,
            )
            if filename.startswith(source['path_prefix']) and filename.endswith('.parquet')
        ]
        random.Random(SEED + source_index).shuffle(files)

        if not files:
            raise RuntimeError(f'No parquet files found for {source["dataset"]} under {source["path_prefix"]}')

        source_plan = source_config_for_plan(source)
        source_plan['revision'] = revision
        source_plan['files'] = files
        plan['sources'][source['name']] = source_plan

    atomic_write_json(PLAN_PATH, plan)
    return plan


def validate_plan(plan):
    ''' Refuse to resume a build with changed corpus settings. '''
    if plan['version'] != 1:
        raise RuntimeError('Unsupported pre-training build plan version.')

    if plan['seed'] != SEED or plan['tokenizer'] != TOKENIZER_NAME:
        raise RuntimeError('Existing build plan does not match the current tokenizer or seed.')

    if plan['total_tokens'] != TOTAL_TOKENS or plan['max_document_tokens'] != MAX_DOCUMENT_TOKENS:
        raise RuntimeError('Existing build plan does not match the current corpus size settings.')

    for source in SOURCES:
        expected = source_config_for_plan(source)
        actual = plan['sources'][source['name']]

        for key, value in expected.items():
            if actual[key] != value:
                raise RuntimeError(f'Existing build plan differs for {source["name"]}: {key}')


def get_plan():
    ''' Load the frozen plan or create it on the first run. '''
    BUILD_PATH.mkdir(parents=True, exist_ok=True)

    if PLAN_PATH.exists():
        plan = json.loads(PLAN_PATH.read_text(encoding='utf-8'))
        validate_plan(plan)
        print(f'Reusing frozen build plan: {PLAN_PATH}')
        return plan

    print('Creating frozen build plan...')
    return create_plan()


def get_local_shard(source, source_plan, filename):
    ''' Download a planned parquet shard once and reuse the local copy. '''
    source_directory = DATA_PATH / source['name']
    local_path = source_directory / filename

    if local_path.exists():
        return local_path

    print(f'Downloading {source["label"]}: {filename}')
    return Path(hf_hub_download(
        repo_id=source['dataset'],
        filename=filename,
        repo_type='dataset',
        revision=source_plan['revision'],
        local_dir=source_directory,
    ))


def filter_batch(source, batch):
    ''' Apply the finalized vectorized filter to one Arrow record batch. '''
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


def make_tokenize_jobs(texts):
    ''' Split filtered documents into multiprocessing-sized tokenizer batches. '''
    for start in range(0, len(texts), TOKENIZE_BATCH_SIZE):
        yield texts[start:start + TOKENIZE_BATCH_SIZE]


def append_tokenized_result(file, tokens, lengths, remaining_tokens):
    ''' Write complete documents, truncating only the final document to hit quota exactly. '''
    offset = 0
    documents = 0
    written_tokens = 0

    for length_value in lengths:
        length = int(length_value)

        if remaining_tokens == 0:
            break

        if length <= remaining_tokens:
            tokens[offset:offset + length].tofile(file)
            offset += length
            remaining_tokens -= length
            written_tokens += length
            documents += 1
            continue

        if remaining_tokens == 1:
            np.asarray([EOS_TOKEN_ID], dtype=np.uint16).tofile(file)
        else:
            final_document = np.empty(remaining_tokens, dtype=np.uint16)
            final_document[:-1] = tokens[offset:offset + remaining_tokens - 1]
            final_document[-1] = EOS_TOKEN_ID
            final_document.tofile(file)

        written_tokens += remaining_tokens
        remaining_tokens = 0
        documents += 1
        break

    return written_tokens, documents, remaining_tokens


def process_shard(source, shard_path, output_path, token_budget, pool, shard_seed):
    ''' Filter and tokenize one local parquet shard into an atomic binary part. '''
    temporary_path = output_path.with_suffix('.bin.tmp')
    temporary_path.unlink(missing_ok=True)
    parquet = pq.ParquetFile(shard_path)
    random_generator = random.Random(shard_seed)
    tokens_written = 0
    documents_written = 0
    rows_seen = 0
    rows_accepted = 0
    progress = tqdm(total=token_budget, unit='tok', unit_scale=True, desc=source['name'])

    with temporary_path.open('wb', buffering=0) as file:
        for batch in parquet.iter_batches(
            batch_size=PARQUET_BATCH_ROWS,
            columns=list(source['columns']),
            use_threads=True,
        ):
            rows_seen += batch.num_rows
            texts = filter_batch(source, batch)
            rows_accepted += len(texts)
            random_generator.shuffle(texts)

            for tokens, lengths in pool.imap(tokenize_text_batch, make_tokenize_jobs(texts), chunksize=1):
                remaining = token_budget - tokens_written
                written, documents, remaining = append_tokenized_result(file, tokens, lengths, remaining)
                tokens_written += written
                documents_written += documents
                progress.update(written)

                if remaining == 0:
                    break

            progress.set_postfix(rows=f'{rows_seen:,}', accepted=f'{rows_accepted:,}')

            if tokens_written == token_budget:
                break

        file.flush()
        os.fsync(file.fileno())

    progress.close()
    temporary_path.replace(output_path)
    return {
        'tokens': tokens_written,
        'documents': documents_written,
        'rows_seen': rows_seen,
        'rows_accepted': rows_accepted,
        'bytes': output_path.stat().st_size,
    }


def part_paths(source, index):
    ''' Return binary and metadata paths for one deterministic source part. '''
    directory = BUILD_PATH / source['name'] / 'parts'
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f'{index:05d}.bin', directory / f'{index:05d}.json'


def remove_incomplete_parts(source):
    ''' Remove crash leftovers that do not have matching metadata. '''
    directory = BUILD_PATH / source['name'] / 'parts'

    if not directory.exists():
        return

    for temporary_path in directory.glob('*.tmp'):
        temporary_path.unlink()

    for binary_path in directory.glob('*.bin'):
        metadata_path = binary_path.with_suffix('.json')

        if not metadata_path.exists():
            binary_path.unlink()


def load_completed_parts(source, source_plan):
    ''' Load and validate completed shard parts for resumable builds. '''
    remove_incomplete_parts(source)
    parts = []

    for index, filename in enumerate(source_plan['files']):
        binary_path, metadata_path = part_paths(source, index)

        if not binary_path.exists() or not metadata_path.exists():
            break

        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))

        if metadata['source_file'] != filename or metadata['revision'] != source_plan['revision']:
            raise RuntimeError(f'Part metadata mismatch: {metadata_path}')

        if binary_path.stat().st_size != metadata['tokens'] * UINT16_BYTES:
            raise RuntimeError(f'Part size mismatch: {binary_path}')

        parts.append((binary_path, metadata_path, metadata))

    return parts


def build_source(source, source_plan, pool):
    ''' Build one source quota from deterministic parquet shards. '''
    completed_parts = load_completed_parts(source, source_plan)
    token_count = sum(part[2]['tokens'] for part in completed_parts)
    document_count = sum(part[2]['documents'] for part in completed_parts)
    target_tokens = source['target_tokens']

    print(f'\n{source["label"]}: {token_count:,} / {target_tokens:,} tokens already complete')

    if token_count >= target_tokens:
        return completed_parts

    for index in range(len(completed_parts), len(source_plan['files'])):
        filename = source_plan['files'][index]
        binary_path, metadata_path = part_paths(source, index)
        remaining = target_tokens - token_count
        shard_path = get_local_shard(source, source_plan, filename)
        print(f'Processing {filename} ({shard_path.stat().st_size / 1024 ** 3:.2f} GiB)')
        statistics = process_shard(
            source,
            shard_path,
            binary_path,
            remaining,
            pool,
            SEED + index + 10_000 * (SOURCES.index(source) + 1),
        )
        metadata = {
            'source': source['name'],
            'dataset': source['dataset'],
            'revision': source_plan['revision'],
            'source_file': filename,
            **statistics,
        }
        atomic_write_json(metadata_path, metadata)
        completed_parts.append((binary_path, metadata_path, metadata))
        token_count += statistics['tokens']
        document_count += statistics['documents']
        print(f'{source["label"]}: {token_count:,} / {target_tokens:,} tokens, {document_count:,} documents')

        if token_count == target_tokens:
            return completed_parts

    raise RuntimeError(f'{source["label"]} exhausted its planned shards before reaching {target_tokens:,} tokens.')


def merge_parts(plan, source_parts):
    ''' Concatenate completed source parts into the final uint16 corpus and manifest. '''
    temporary_path = OUTPUT_PATH.with_suffix('.bin.tmp')
    temporary_path.unlink(missing_ok=True)
    digest = hashlib.sha256()
    manifest_sources = {}
    total_tokens = 0

    with temporary_path.open('wb') as output_file:
        for source in SOURCES:
            parts = source_parts[source['name']]
            source_tokens = 0
            source_documents = 0

            for binary_path, metadata_path, metadata in parts:
                with binary_path.open('rb') as input_file:
                    while chunk := input_file.read(COPY_BUFFER_SIZE):
                        output_file.write(chunk)
                        digest.update(chunk)

                source_tokens += metadata['tokens']
                source_documents += metadata['documents']

            manifest_sources[source['name']] = {
                'label': source['label'],
                'dataset': source['dataset'],
                'revision': plan['sources'][source['name']]['revision'],
                'filter': source['filter'],
                'target_tokens': source['target_tokens'],
                'tokens': source_tokens,
                'documents': source_documents,
                'parts': len(parts),
            }
            total_tokens += source_tokens

        output_file.flush()
        os.fsync(output_file.fileno())

    if total_tokens != TOTAL_TOKENS:
        raise RuntimeError(f'Final token count is {total_tokens:,}, expected {TOTAL_TOKENS:,}.')

    if temporary_path.stat().st_size != TOTAL_TOKENS * UINT16_BYTES:
        raise RuntimeError('Final binary size does not match uint16 token count.')

    temporary_path.replace(OUTPUT_PATH)
    manifest = {
        'version': 1,
        'file': OUTPUT_PATH.name,
        'sha256': digest.hexdigest(),
        'dtype': 'uint16',
        'bytes_per_token': UINT16_BYTES,
        'file_size_bytes': OUTPUT_PATH.stat().st_size,
        'tokens': total_tokens,
        'tokenizer': TOKENIZER_NAME,
        'vocab_size': 50257,
        'eos_token_id': EOS_TOKEN_ID,
        'max_document_tokens': MAX_DOCUMENT_TOKENS,
        'seed': SEED,
        'workers': WORKER_COUNT,
        'sources': manifest_sources,
    }
    atomic_write_json(MANIFEST_PATH, manifest)
    return manifest


def main():
    ''' Build the finalized 460M-token pre-training corpus. '''
    if sum(source['target_tokens'] for source in SOURCES) != TOTAL_TOKENS:
        raise RuntimeError('Source token quotas do not add up to TOTAL_TOKENS.')

    DATA_PATH.mkdir(parents=True, exist_ok=True)
    BUILD_PATH.mkdir(parents=True, exist_ok=True)
    print(f'CPU tokenizer workers: {WORKER_COUNT}')
    print(f'Final corpus: {TOTAL_TOKENS:,} uint16 tokens ({TOTAL_TOKENS * UINT16_BYTES / 1024 ** 3:.2f} GiB)')
    print('GPUs are intentionally unused during corpus construction.')
    AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    plan = get_plan()
    context = multiprocessing.get_context('spawn')
    source_parts = {}

    with context.Pool(WORKER_COUNT, initializer=initialize_worker) as pool:
        for source in SOURCES:
            source_parts[source['name']] = build_source(source, plan['sources'][source['name']], pool)

    print('\nMerging finalized source parts...')
    manifest = merge_parts(plan, source_parts)
    print(f'Wrote {OUTPUT_PATH} ({manifest["file_size_bytes"] / 1024 ** 3:.2f} GiB)')
    print(f'Wrote {MANIFEST_PATH}')
    print(f'SHA-256: {manifest["sha256"]}')


if __name__ == '__main__':
    main()
