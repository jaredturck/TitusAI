''' Compare the strongest local pre-training dataset slices. '''

import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean, median

from datasets import load_dataset

DATA_PATH = Path(__file__).parent / 'data'
OUTPUT_PATH = Path(__file__).parent / 'output'
OUTPUT_FILE = OUTPUT_PATH / 'final_sweep.md'
SAMPLE_COUNT = 20
MAX_TEXT_CHARACTERS = 4000
SEED = 42

SLICES = (
    {
        'name': 'Cosmopedia v2 middle-school textbooks',
        'dataset': 'cosmopedia_v2',
        'filter': 'cosmopedia_middle_school',
        'metadata_fields': ('seed_data', 'format', 'audience', 'token_length'),
    },
    {
        'name': 'Cosmopedia v2 general textbooks',
        'dataset': 'cosmopedia_v2',
        'filter': 'cosmopedia_general',
        'metadata_fields': ('seed_data', 'format', 'audience', 'token_length'),
    },
    {
        'name': 'DCLM-Edu high-quality natural text',
        'dataset': 'dclm_edu',
        'filter': 'dclm_high_quality',
        'metadata_fields': ('edu_int_score', 'edu_score', 'language_score', 'url'),
    },
    {
        'name': 'FineWeb-Edu high-quality natural text',
        'dataset': 'fineweb_edu',
        'filter': 'fineweb_high_quality',
        'metadata_fields': ('int_score', 'score', 'language_score', 'token_count', 'url'),
    },
)

def get_shard(name):
    ''' Find the existing local parquet shard for a dataset. '''
    files = sorted((DATA_PATH / name).rglob('*.parquet'))
    return files[0]

def load_shard(name):
    ''' Load one existing local parquet shard. '''
    path = get_shard(name)
    print(f'Loading {name}: {path}')
    dataset = load_dataset('parquet', data_files=str(path), split='train')
    print(f'Loaded {len(dataset):,} rows.')
    return dataset

def matches(row, filter_name):
    ''' Return whether one row belongs to a candidate slice. '''
    if filter_name == 'cosmopedia_middle_school':
        return row['audience'] == 'middle_school_students' and row['format'].startswith('textbook')

    if filter_name == 'cosmopedia_general':
        return row['audience'] == 'general' and row['format'].startswith('textbook')

    if filter_name == 'dclm_high_quality':
        return row['edu_int_score'] >= 4 and row['language_score'] >= 0.95 and 500 <= len(row['text']) <= 8000

    return row['int_score'] >= 4 and row['language_score'] >= 0.95 and 100 <= row['token_count'] <= 1024

def get_metadata(row, fields):
    ''' Keep metadata useful for manual comparison. '''
    return {field: row[field] for field in fields if field in row}

def collect_slice(dataset, slice_config):
    ''' Count and sample one candidate slice. '''
    random_generator = random.Random(SEED)
    samples = []
    match_count = 0

    for row in dataset:
        if not matches(row, slice_config['filter']):
            continue

        text = row['text'].strip()

        if not text:
            continue

        match_count += 1
        sample = {
            'characters': len(text),
            'metadata': get_metadata(row, slice_config['metadata_fields']),
            'text': text[:MAX_TEXT_CHARACTERS],
            'truncated': len(text) > MAX_TEXT_CHARACTERS,
        }

        if len(samples) < SAMPLE_COUNT:
            samples.append(sample)
        else:
            index = random_generator.randrange(match_count)

            if index < SAMPLE_COUNT:
                samples[index] = sample

    return match_count, samples

def count_cosmopedia_metadata(dataset):
    ''' Count Cosmopedia v2 audience, format, and seed distributions. '''
    audiences = Counter()
    formats = Counter()
    seeds = Counter()

    for row in dataset:
        audiences[row['audience']] += 1
        formats[row['format']] += 1
        seeds[row['seed_data']] += 1

    return audiences, formats, seeds

def write_counter(lines, title, counter):
    ''' Append the most common metadata values to the report. '''
    lines.extend([
        f'### {title}',
        '',
        '| Value | Rows |',
        '| --- | ---: |',
    ])

    for value, count in counter.most_common(20):
        lines.append(f'| `{value}` | {count:,} |')

    lines.append('')

def write_slice(lines, slice_config, total_rows, match_count, samples):
    ''' Append one candidate slice and its samples to the report. '''
    characters = [sample['characters'] for sample in samples]
    lines.extend([
        f'## {slice_config["name"]}',
        '',
        f'Matching rows in local shard: {match_count:,} / {total_rows:,} ({match_count / total_rows:.1%})',
        f'Sampled documents: {len(samples)}',
        f'Average sampled characters: {mean(characters):,.0f}',
        f'Median sampled characters: {median(characters):,.0f}',
        '',
    ])

    for index, sample in enumerate(samples, 1):
        lines.extend([
            f'### Sample {index}',
            '',
            f'- Characters: {sample["characters"]:,}',
            f'- Truncated: {sample["truncated"]}',
            f'- Metadata: `{json.dumps(sample["metadata"], ensure_ascii=False)}`',
            '',
            '\n'.join(f'    {line}' for line in sample['text'].splitlines()),
            '',
        ])

def main():
    ''' Build the final local-only dataset comparison report. '''
    OUTPUT_PATH.mkdir(exist_ok=True)
    datasets = {
        'cosmopedia_v2': load_shard('cosmopedia_v2'),
        'dclm_edu': load_shard('dclm_edu'),
        'fineweb_edu': load_shard('fineweb_edu'),
    }
    lines = [
        '# Final pre-training slice sweep',
        '',
        'This report compares filtered slices from the strongest candidate datasets using only already-downloaded local shards.',
        '',
        '# Cosmopedia v2 metadata distribution',
        '',
    ]
    audiences, formats, seeds = count_cosmopedia_metadata(datasets['cosmopedia_v2'])
    write_counter(lines, 'Audiences', audiences)
    write_counter(lines, 'Formats', formats)
    write_counter(lines, 'Seed data', seeds)

    for slice_config in SLICES:
        dataset = datasets[slice_config['dataset']]
        print(f'Collecting {slice_config["name"]}...')
        match_count, samples = collect_slice(dataset, slice_config)
        print(f'Found {match_count:,} matching rows.')
        write_slice(lines, slice_config, len(dataset), match_count, samples)

    OUTPUT_FILE.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {OUTPUT_FILE}')

if __name__ == '__main__':
    main()
