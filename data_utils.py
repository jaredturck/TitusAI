import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path

import numpy as np


CONTROL_CHARACTERS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
PARAGRAPH_SPLIT = re.compile(r'\n\s*\n')


def normalize_document(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = CONTROL_CHARACTERS.sub('', text)
    return text.strip('\n')


def stable_hash(value):
    return hashlib.blake2b(
        value.encode('utf-8'),
        digest_size=16,
    ).hexdigest()


def deterministic_split(document_id, validation_fraction):
    digest = hashlib.blake2b(
        document_id.encode('utf-8'),
        digest_size=8,
    ).digest()
    value = int.from_bytes(digest, byteorder='big') / 2**64
    return 'validation' if value < validation_fraction else 'train'


def extract_first(record, fields):
    for field in fields:
        value = record.get(field)
        if value is not None and value != '':
            return value
    return None


def build_document_id(record, source, id_fields, text):
    values = []
    for field in id_fields:
        value = record.get(field)
        if value is not None and value != '':
            values.append(str(value))

    if values:
        return f'{source}:' + ':'.join(values)

    return f'{source}:{stable_hash(text)}'


class DeduplicationStore:
    def __init__(self, database_path, paragraph_minimum_characters=200):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.paragraph_minimum_characters = paragraph_minimum_characters
        self.connection = sqlite3.connect(self.database_path)
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('PRAGMA synchronous=NORMAL')
        self.connection.execute(
            'CREATE TABLE IF NOT EXISTS documents ('
            'hash TEXT PRIMARY KEY'
            ')'
        )
        self.connection.execute(
            'CREATE TABLE IF NOT EXISTS paragraphs ('
            'hash TEXT PRIMARY KEY'
            ')'
        )
        self.pending_writes = 0

    def is_duplicate_document(self, text):
        document_hash = stable_hash(text)
        cursor = self.connection.execute(
            'INSERT OR IGNORE INTO documents(hash) VALUES (?)',
            (document_hash,),
        )
        self.record_write()
        return cursor.rowcount == 0

    def remove_duplicate_paragraphs(self, text):
        paragraphs = PARAGRAPH_SPLIT.split(text)
        kept_paragraphs = []

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(paragraph) < self.paragraph_minimum_characters:
                kept_paragraphs.append(paragraph)
                continue

            paragraph_hash = stable_hash(paragraph)
            cursor = self.connection.execute(
                'INSERT OR IGNORE INTO paragraphs(hash) VALUES (?)',
                (paragraph_hash,),
            )
            self.record_write()

            if cursor.rowcount > 0:
                kept_paragraphs.append(paragraph)

        return '\n\n'.join(kept_paragraphs)

    def record_write(self):
        self.pending_writes += 1
        if self.pending_writes >= 1000:
            self.connection.commit()
            self.pending_writes = 0

    def close(self):
        self.connection.commit()
        self.connection.close()


class ShardWriter:
    def __init__(self, output_path, source_name, sequence_length, sequences_per_shard, store_loss_mask=True, existing_shards=None):
        self.output_path = Path(output_path)
        self.source_name = source_name
        self.sequence_length = sequence_length
        self.stored_sequence_length = sequence_length + 1
        self.sequences_per_shard = sequences_per_shard
        self.store_loss_mask = store_loss_mask
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.token_sequences = []
        self.segment_sequences = []
        self.loss_mask_sequences = []
        self.shards = list(existing_shards or [])
        self.total_sequences = sum(
            shard['num_sequences']
            for shard in self.shards
        )
        self.shard_index = len(self.shards)

    def add_sequence(self, token_ids, segment_ids, loss_mask):
        assert len(token_ids) == self.stored_sequence_length
        assert len(segment_ids) == self.stored_sequence_length
        assert len(loss_mask) == self.stored_sequence_length
        assert max(token_ids) < 65_536
        assert max(segment_ids) < 65_536

        self.token_sequences.append(np.asarray(token_ids, dtype=np.uint16))
        self.segment_sequences.append(np.asarray(segment_ids, dtype=np.uint16))
        if self.store_loss_mask:
            self.loss_mask_sequences.append(np.asarray(loss_mask, dtype=np.uint8))

        if len(self.token_sequences) >= self.sequences_per_shard:
            self.flush()

    def write_array(self, array, path):
        temporary_path = Path(f'{path}.writing')
        array.tofile(temporary_path)
        os.replace(temporary_path, path)

    def flush(self):
        if not self.token_sequences:
            return

        token_array = np.stack(self.token_sequences)
        segment_array = np.stack(self.segment_sequences)
        loss_mask_array = None
        if self.store_loss_mask:
            loss_mask_array = np.stack(self.loss_mask_sequences)

        shard_prefix = f'{self.source_name}_{self.shard_index:06d}'
        token_file = f'{shard_prefix}.tokens.bin'
        segment_file = f'{shard_prefix}.segments.bin'
        loss_mask_file = None

        self.write_array(token_array, self.output_path / token_file)
        self.write_array(segment_array, self.output_path / segment_file)

        if loss_mask_array is not None:
            loss_mask_file = f'{shard_prefix}.loss.bin'
            self.write_array(loss_mask_array, self.output_path / loss_mask_file)

        sequence_count = token_array.shape[0]
        self.shards.append({
            'source': self.source_name,
            'tokens': token_file,
            'segments': segment_file,
            'loss_mask': loss_mask_file,
            'num_sequences': sequence_count,
        })
        self.total_sequences += sequence_count
        self.shard_index += 1
        self.token_sequences.clear()
        self.segment_sequences.clear()
        self.loss_mask_sequences.clear()

    def close(self):
        self.flush()
        return self.shards


class SequencePacker:
    def __init__(self, writer, sequence_length):
        self.writer = writer
        self.sequence_length = sequence_length
        self.stored_sequence_length = sequence_length + 1
        self.token_buffer = []
        self.segment_buffer = []
        self.loss_mask_buffer = []
        self.next_segment_id = 0
        self.documents = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def add_document(self, token_ids, loss_mask=None):
        if loss_mask is None:
            loss_mask = [1] * len(token_ids)

        assert len(token_ids) == len(loss_mask)

        segment_id = self.next_segment_id
        self.next_segment_id += 1
        self.documents += 1
        self.input_tokens += len(token_ids)
        self.token_buffer.extend(token_ids)
        self.segment_buffer.extend([segment_id] * len(token_ids))
        self.loss_mask_buffer.extend(loss_mask)
        self.emit_sequences()

    def remap_segments(self, segment_ids):
        mapping = {}
        remapped = []

        for segment_id in segment_ids:
            if segment_id not in mapping:
                mapping[segment_id] = len(mapping)
            remapped.append(mapping[segment_id])

        return remapped

    def emit_sequences(self):
        while len(self.token_buffer) >= self.stored_sequence_length:
            token_ids = self.token_buffer[:self.stored_sequence_length]
            segment_ids = self.segment_buffer[:self.stored_sequence_length]
            loss_mask = self.loss_mask_buffer[:self.stored_sequence_length]
            segment_ids = self.remap_segments(segment_ids)

            self.writer.add_sequence(token_ids, segment_ids, loss_mask)
            self.output_tokens += self.sequence_length

            del self.token_buffer[:self.sequence_length]
            del self.segment_buffer[:self.sequence_length]
            del self.loss_mask_buffer[:self.sequence_length]

    def close(self):
        discarded_tokens = len(self.token_buffer)
        shards = self.writer.close()
        return {
            'documents': self.documents,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'discarded_tokens': discarded_tokens,
            'num_sequences': self.writer.total_sequences,
            'shards': shards,
        }


def remove_stale_writing_files(output_path):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    removed = []

    for path in sorted(output_path.glob('*.writing')):
        path.unlink()
        removed.append(path.name)

    return removed


def recover_source_shards(output_path, source_name, sequence_length, store_loss_mask=False):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    stored_sequence_length = sequence_length + 1
    token_bytes_per_sequence = stored_sequence_length * np.dtype(np.uint16).itemsize
    segment_bytes_per_sequence = stored_sequence_length * np.dtype(np.uint16).itemsize
    loss_bytes_per_sequence = stored_sequence_length * np.dtype(np.uint8).itemsize
    pattern = re.compile(
        rf'^{re.escape(source_name)}_(\d{{6}})\.(tokens|segments|loss)\.bin$'
    )
    files = {}

    for path in output_path.iterdir():
        match = pattern.match(path.name)
        if match is None:
            continue

        shard_index = int(match.group(1))
        file_type = match.group(2)
        files.setdefault(shard_index, {})[file_type] = path

    if not files:
        return []

    expected_indices = list(range(max(files) + 1))
    if sorted(files) != expected_indices:
        raise RuntimeError(
            f'Non-contiguous shard numbering for {source_name} in {output_path}'
        )

    shards = []
    for shard_index in expected_indices:
        shard_files = files[shard_index]
        token_path = shard_files.get('tokens')
        segment_path = shard_files.get('segments')
        loss_path = shard_files.get('loss')

        if token_path is None or segment_path is None:
            raise RuntimeError(
                f'Incomplete shard {source_name}_{shard_index:06d} in {output_path}'
            )

        token_size = token_path.stat().st_size
        segment_size = segment_path.stat().st_size
        if token_size == 0 or token_size % token_bytes_per_sequence != 0:
            raise RuntimeError(f'Invalid token shard size: {token_path}')

        num_sequences = token_size // token_bytes_per_sequence
        if segment_size != num_sequences * segment_bytes_per_sequence:
            raise RuntimeError(
                f'Token and segment shard sizes do not match for {token_path.name}'
            )

        if store_loss_mask and loss_path is None:
            raise RuntimeError(f'Missing loss-mask shard for {token_path.name}')

        if loss_path is not None and loss_path.stat().st_size != num_sequences * loss_bytes_per_sequence:
            raise RuntimeError(f'Invalid loss-mask shard size: {loss_path}')

        shards.append({
            'source': source_name,
            'tokens': token_path.name,
            'segments': segment_path.name,
            'loss_mask': loss_path.name if loss_path is not None else None,
            'num_sequences': num_sequences,
        })

    return shards


def write_manifest(output_path, tokenizer_metadata, sequence_length, source_statistics, shards):
    output_path = Path(output_path)
    manifest = {
        'format_version': 1,
        'sequence_length': sequence_length,
        'stored_sequence_length': sequence_length + 1,
        'token_dtype': 'uint16',
        'segment_dtype': 'uint16',
        'loss_mask_dtype': 'uint8',
        'tokenizer': tokenizer_metadata,
        'sources': source_statistics,
        'shards': shards,
        'num_sequences': sum(shard['num_sequences'] for shard in shards),
    }

    temporary_path = output_path / 'manifest.json.writing'
    final_path = output_path / 'manifest.json'
    temporary_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    os.replace(temporary_path, final_path)
    return manifest


def load_manifest(dataset_path):
    manifest_path = Path(dataset_path) / 'manifest.json'
    return json.loads(manifest_path.read_text(encoding='utf-8'))
