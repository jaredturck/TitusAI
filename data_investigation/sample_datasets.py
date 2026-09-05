''' Stream small samples from candidate pre-training datasets. '''

import json
from pathlib import Path
from statistics import mean, median

from datasets import load_dataset
from transformers import AutoTokenizer

OUTPUT_PATH = Path(__file__).parent / 'output'
TOKENIZER_NAME = 'gpt2'
SAMPLE_COUNT = 20
SHUFFLE_BUFFER_SIZE = 200
MAX_TEXT_CHARACTERS = 5000
SEED = 42

SOURCES = (
    {
        'name': 'DCLM-Edu score 3+',
        'dataset': 'HuggingFaceTB/dclm-edu',
        'config': None,
        'text_field': 'text',
        'score_field': 'edu_int_score',
        'minimum_score': 3,
        'metadata_fields': ('edu_int_score', 'edu_score', 'language_score', 'url'),
    },
    {
        'name': 'FineWeb-Edu score 3+',
        'dataset': 'HuggingFaceFW/fineweb-edu',
        'config': 'sample-10BT',
        'text_field': 'text',
        'score_field': 'int_score',
        'minimum_score': 3,
        'metadata_fields': ('int_score', 'score', 'language_score', 'token_count', 'url'),
    },
    {
        'name': 'Cosmopedia 100k',
        'dataset': 'HuggingFaceTB/cosmopedia-100k',
        'config': None,
        'text_field': 'text',
        'score_field': None,
        'minimum_score': None,
        'metadata_fields': ('seed_data', 'format', 'audience', 'text_token_length'),
    },
    {
        'name': 'TinyStories V2 GPT-4',
        'dataset': 'maveriq/tinystoriesv2_gpt4',
        'config': None,
        'text_field': 'text',
        'score_field': None,
        'minimum_score': None,
        'metadata_fields': (),
    },
    {
        'name': 'ClimbMix shuffled',
        'dataset': 'karpathy/climbmix-400b-shuffle',
        'config': None,
        'text_field': 'text',
        'score_field': None,
        'minimum_score': None,
        'metadata_fields': (),
    },
)

def load_source(source):
    ''' Load one dataset as a streaming iterable. '''
    if source['config']:
        dataset = load_dataset(source['dataset'], source['config'], split='train', streaming=True)
    else:
        dataset = load_dataset(source['dataset'], split='train', streaming=True)

    return dataset.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER_SIZE)

def get_metadata(row, source):
    ''' Keep only metadata useful for manual comparison. '''
    return {field: row[field] for field in source['metadata_fields']}

def collect_samples(source, tokenizer):
    ''' Collect a small filtered sample from one streaming dataset. '''
    samples = []
    dataset = load_source(source)

    for row in dataset:
        if source['score_field'] and row[source['score_field']] < source['minimum_score']:
            continue

        text = row[source['text_field']].strip()

        if not text:
            continue

        preview = text[:MAX_TEXT_CHARACTERS]
        preview_ids = tokenizer(preview, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False, verbose=False)['input_ids']
        samples.append({
            'dataset': source['name'],
            'source': source['dataset'],
            'characters': len(text),
            'preview_tokens': len(preview_ids),
            'truncated': len(text) > len(preview),
            'metadata': get_metadata(row, source),
            'text': preview,
        })

        if len(samples) == SAMPLE_COUNT:
            break

    return samples

def indent_text(text):
    ''' Format sample text as an indented Markdown block. '''
    return '\n'.join(f'    {line}' for line in text.splitlines())

def write_dataset_report(source, samples):
    ''' Write one human-readable Markdown report. '''
    filename = source['name'].lower().replace(' ', '_').replace('+', 'plus') + '.md'
    path = OUTPUT_PATH / filename
    lines = [
        f'# {source["name"]}',
        '',
        f'Source: `{source["dataset"]}`',
        f'Samples: {len(samples)}',
        '',
    ]

    if source['score_field']:
        lines.extend([
            f'Filter: `{source["score_field"]} >= {source["minimum_score"]}`',
            '',
        ])

    for index, sample in enumerate(samples, 1):
        lines.extend([
            f'## Sample {index}',
            '',
            f'- Original characters: {sample["characters"]:,}',
            f'- Preview GPT-2 tokens: {sample["preview_tokens"]:,}',
            f'- Truncated: {sample["truncated"]}',
            f'- Metadata: `{json.dumps(sample["metadata"], ensure_ascii=False)}`',
            '',
            indent_text(sample['text']),
            '',
        ])

    path.write_text('\n'.join(lines), encoding='utf-8')
    return path

def write_summary(results):
    ''' Write compact statistics and one combined JSONL file. '''
    summary_path = OUTPUT_PATH / 'summary.md'
    jsonl_path = OUTPUT_PATH / 'samples.jsonl'
    lines = [
        '# Dataset sample summary',
        '',
        '| Dataset | Samples | Avg chars | Median chars | Avg preview tokens | Truncated |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]

    with jsonl_path.open('w', encoding='utf-8') as file:
        for source, samples in results:
            characters = [sample['characters'] for sample in samples]
            preview_tokens = [sample['preview_tokens'] for sample in samples]
            truncated = sum(sample['truncated'] for sample in samples)
            lines.append(f'| {source["name"]} | {len(samples)} | {mean(characters):,.0f} | {median(characters):,.0f} | {mean(preview_tokens):,.0f} | {truncated} |')

            for sample in samples:
                file.write(json.dumps(sample, ensure_ascii=False) + '\n')

    summary_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary_path, jsonl_path

def main():
    ''' Sample all candidate datasets and write inspection files. '''
    OUTPUT_PATH.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.model_max_length = 1_000_000_000
    results = []

    for source in SOURCES:
        print(f'Sampling {source["name"]}...')
        samples = collect_samples(source, tokenizer)
        report_path = write_dataset_report(source, samples)
        results.append((source, samples))
        print(f'Wrote {report_path}')

    summary_path, jsonl_path = write_summary(results)
    print(f'Wrote {summary_path}')
    print(f'Wrote {jsonl_path}')

if __name__ == '__main__':
    main()
