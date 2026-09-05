''' Compare educational quality score bands in downloaded web datasets. '''

import random
from pathlib import Path
from statistics import mean

from datasets import load_dataset

DATA_PATH = Path(__file__).parent / 'data'
OUTPUT_PATH = Path(__file__).parent / 'output'
SAMPLE_COUNT = 12
MAX_TEXT_CHARACTERS = 2500
SEED = 123

SOURCES = (
    {
        'name': 'DCLM-Edu',
        'local_name': 'dclm_edu',
        'score_field': 'edu_int_score',
        'float_score_field': 'edu_score',
        'language_score_field': 'language_score',
    },
    {
        'name': 'FineWeb-Edu',
        'local_name': 'fineweb_edu',
        'score_field': 'int_score',
        'float_score_field': 'score',
        'language_score_field': 'language_score',
    },
)

def get_shard(source):
    ''' Find the already-downloaded parquet shard. '''
    files = sorted((DATA_PATH / source['local_name']).rglob('*.parquet'))
    return files[0]

def load_shard(source):
    ''' Load one local parquet shard. '''
    path = get_shard(source)
    print(f'Loading {source["name"]}: {path}')
    dataset = load_dataset('parquet', data_files=str(path), split='train')
    print(f'Loaded {len(dataset):,} rows.')
    return path, dataset

def get_score_indices(dataset, score_field):
    ''' Group local row indices by exact educational score. '''
    grouped = {}

    for index, score in enumerate(dataset[score_field]):
        grouped.setdefault(score, []).append(index)

    return grouped

def sample_score(dataset, source, indices, score):
    ''' Sample readable examples from one exact score band. '''
    random_generator = random.Random(SEED + score)
    chosen = random_generator.sample(indices, min(SAMPLE_COUNT, len(indices)))
    samples = []

    for index in chosen:
        row = dataset[index]
        text = row['text'].strip()
        samples.append({
            'characters': len(text),
            'score': row[source['float_score_field']],
            'language_score': row[source['language_score_field']],
            'url': row.get('url', ''),
            'text': text[:MAX_TEXT_CHARACTERS],
            'truncated': len(text) > MAX_TEXT_CHARACTERS,
        })

    return samples

def indent_text(text):
    ''' Format text as an indented Markdown block. '''
    return '\n'.join(f'    {line}' for line in text.splitlines())

def write_source(lines, source, path, dataset):
    ''' Add one dataset score sweep to the report. '''
    grouped = get_score_indices(dataset, source['score_field'])
    total = len(dataset)
    lines.extend([
        f'# {source["name"]}',
        '',
        f'Shard: `{path}`',
        f'Rows: {total:,}',
        '',
        '| Exact score | Rows | Share |',
        '| ---: | ---: | ---: |',
    ])

    for score in sorted(grouped):
        count = len(grouped[score])
        lines.append(f'| {score} | {count:,} | {count / total:.1%} |')

    lines.append('')

    for score in (3, 4, 5):
        if score not in grouped:
            continue

        samples = sample_score(dataset, source, grouped[score], score)
        lines.extend([
            f'## Exact score {score}',
            '',
            f'Sampled {len(samples)} of {len(grouped[score]):,} rows.',
            f'Average sampled characters: {mean(sample["characters"] for sample in samples):,.0f}',
            f'Average sampled language score: {mean(sample["language_score"] for sample in samples):.3f}',
            '',
        ])

        for index, sample in enumerate(samples, 1):
            lines.extend([
                f'### Sample {score}.{index}',
                '',
                f'- Characters: {sample["characters"]:,}',
                f'- Educational score: {sample["score"]:.3f}',
                f'- Language score: {sample["language_score"]:.3f}',
                f'- Truncated: {sample["truncated"]}',
                f'- URL: {sample["url"]}',
                '',
                indent_text(sample['text']),
                '',
            ])

def main():
    ''' Write score-band samples from local web shards. '''
    OUTPUT_PATH.mkdir(exist_ok=True)
    lines = ['# Educational score quality sweep', '']

    for source in SOURCES:
        path, dataset = load_shard(source)
        write_source(lines, source, path, dataset)

    path = OUTPUT_PATH / 'quality_sweep.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {path}')

if __name__ == '__main__':
    main()
