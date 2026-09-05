''' Verify the generated TitusAI pre-training corpus. '''

import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT_PATH = Path(__file__).parent
BUILD_PATH = ROOT_PATH / 'pretrain_corpus'
CORPUS_PATH = BUILD_PATH / 'pretrain.bin'
MANIFEST_PATH = BUILD_PATH / 'pretrain.json'
CHUNK_TOKENS = 16_777_216
EOS_TOKEN_ID = 50256
VOCAB_SIZE = 50257


def file_sha256(path):
    ''' Hash a file without loading it into memory. '''
    digest = hashlib.sha256()

    with path.open('rb') as file:
        while chunk := file.read(16 * 1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    ''' Validate corpus size, token range, EOS count, quotas, and checksum. '''
    manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    expected_tokens = manifest['tokens']
    expected_bytes = expected_tokens * manifest['bytes_per_token']

    if CORPUS_PATH.stat().st_size != expected_bytes:
        raise RuntimeError('Corpus byte size does not match the manifest.')

    source_tokens = sum(source['tokens'] for source in manifest['sources'].values())

    if source_tokens != expected_tokens:
        raise RuntimeError('Source token counts do not add up to the final token count.')

    corpus = np.memmap(CORPUS_PATH, dtype=np.uint16, mode='r')
    minimum_token = VOCAB_SIZE
    maximum_token = 0
    eos_count = 0

    with tqdm(total=len(corpus), unit='tok', unit_scale=True, desc='Verifying tokens') as progress:
        for start in range(0, len(corpus), CHUNK_TOKENS):
            chunk = corpus[start:start + CHUNK_TOKENS]
            minimum_token = min(minimum_token, int(chunk.min()))
            maximum_token = max(maximum_token, int(chunk.max()))
            eos_count += int(np.count_nonzero(chunk == EOS_TOKEN_ID))
            progress.update(len(chunk))

    if maximum_token >= VOCAB_SIZE:
        raise RuntimeError(f'Invalid GPT-2 token ID found: {maximum_token}')

    expected_documents = sum(source['documents'] for source in manifest['sources'].values())

    if eos_count != expected_documents:
        raise RuntimeError(f'EOS count {eos_count:,} does not match document count {expected_documents:,}.')

    checksum = file_sha256(CORPUS_PATH)

    if checksum != manifest['sha256']:
        raise RuntimeError('Corpus SHA-256 does not match the manifest.')

    print(f'Tokens: {len(corpus):,}')
    print(f'Documents: {eos_count:,}')
    print(f'Token range: {minimum_token:,}..{maximum_token:,}')
    print(f'File size: {CORPUS_PATH.stat().st_size / 1024 ** 3:.2f} GiB')
    print(f'SHA-256: {checksum}')
    print('Corpus verification passed.')


if __name__ == '__main__':
    main()
