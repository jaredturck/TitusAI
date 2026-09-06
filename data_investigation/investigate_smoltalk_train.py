''' Inspect the Smol-SmolTalk training split without downloading it. '''

import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from urllib.parse import urlencode
from urllib.request import urlopen

from transformers import AutoTokenizer

DATASET_NAME = 'HuggingFaceTB/smol-smoltalk'
TOKENIZER_NAME = 'gpt2'
OUTPUT_PATH = Path('data_investigation/output/smol_smoltalk_train_report.txt')
SAMPLE_SIZE = 200
SAMPLE_CHARACTER_LIMIT = 1200

SOURCES = [
    'smol-magpie-ultra-short',
    'self-oss-instruct',
    'openhermes-50k',
    'smol-contraints',
    'smollm-rewrite-30k',
    'smol-summarize-20k',
    'smol-summarize-5k',
    'explore-instruct-rewrite',
    'longalign',
    'everyday-conversations',
]

CODE_MARKERS = ('```', 'python', 'function', 'code', 'program', 'algorithm')
CONSTRAINT_MARKERS = ('exactly', 'must contain', 'response should', 'bullet points', 'all lowercase', 'all uppercase')

def fetch_source(source, offset, length):
    ''' Fetch filtered rows from the Hugging Face dataset viewer. '''
    query = urlencode({
        'dataset': DATASET_NAME,
        'config': 'default',
        'split': 'train',
        'where': f'"source"=\'{source}\'',
        'offset': offset,
        'length': length,
    })
    with urlopen(f'https://datasets-server.huggingface.co/filter?{query}') as response:
        return json.load(response)

def get_samples(source):
    ''' Get exact source count and a small spread of training rows. '''
    first = fetch_source(source, 0, min(100, SAMPLE_SIZE))
    count = first['num_rows_total']
    rows = first['rows']

    if count > 100 and SAMPLE_SIZE > 100:
        offset = max(0, count // 2 - 50)
        rows += fetch_source(source, offset, min(100, SAMPLE_SIZE - 100))['rows']

    return count, rows

def main():
    ''' Write a compact train-split suitability report. '''
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    source_data = {}
    total_rows = 0

    for source in SOURCES:
        print(f'Inspecting {source}')
        count, rows = get_samples(source)
        source_data[source] = (count, rows)
        total_rows += count

    report = []
    report.append('Smol-SmolTalk training split investigation')
    report.append('=========================================')
    report.append(f'Dataset: {DATASET_NAME}')
    report.append(f'Exact rows across known sources: {total_rows:,}')
    report.append(f'Sampled rows per source: up to {SAMPLE_SIZE}')
    report.append('Rows are queried through the Hugging Face dataset viewer; the full training dataset is not downloaded.')
    report.append('Token counts use raw message content only and exclude role labels, separators, and EOS tokens.')
    report.append('')
    report.append('Source counts and sampled characteristics')
    report.append('-----------------------------------------')
    report.append('source | rows | share | sampled | mean tokens | median | <=1024 | multi-turn | system | code-like | constraint-like')

    detailed = {}

    for source in SOURCES:
        count, rows = source_data[source]
        token_counts = []
        multi_turn = 0
        system = 0
        code_like = 0
        constraint_like = 0
        role_patterns = Counter()

        for item in rows:
            messages = item['row']['messages']
            roles = [message['role'] for message in messages]
            contents = [message['content'] or '' for message in messages]
            text = '\n'.join(contents)
            user_text = '\n'.join(message['content'] or '' for message in messages if message['role'] == 'user').lower()
            token_counts.append(len(tokenizer.encode(text, add_special_tokens=False)))
            role_patterns[' > '.join(roles)] += 1

            if len(messages) > 2:
                multi_turn += 1
            if 'system' in roles:
                system += 1
            if any(marker in user_text for marker in CODE_MARKERS):
                code_like += 1
            if any(marker in user_text for marker in CONSTRAINT_MARKERS):
                constraint_like += 1

        sampled = len(rows)
        fit = sum(value <= 1024 for value in token_counts) / sampled
        report.append(
            f'{source} | {count:,} | {count / total_rows:.1%} | {sampled} | {mean(token_counts):.1f} | '
            f'{median(token_counts):.1f} | {fit:.1%} | {multi_turn / sampled:.1%} | {system / sampled:.1%} | '
            f'{code_like / sampled:.1%} | {constraint_like / sampled:.1%}'
        )
        detailed[source] = (rows, role_patterns)

    report.append('')
    report.append('Representative training conversations')
    report.append('-------------------------------------')

    for source in SOURCES:
        rows, role_patterns = detailed[source]
        report.append('')
        report.append(f'### {source}')
        report.append('Sampled role patterns: ' + ', '.join(f'{pattern} ({count})' for pattern, count in role_patterns.most_common(4)))

        sample_indexes = sorted(set((0, len(rows) // 2, len(rows) - 1)))
        for sample_index in sample_indexes:
            item = rows[sample_index]
            report.append(f'Row {item["row_idx"]:,}')
            for message in item['row']['messages']:
                content = (message['content'] or '').strip()
                if len(content) > SAMPLE_CHARACTER_LIMIT:
                    content = content[:SAMPLE_CHARACTER_LIMIT] + ' [...]'
                report.append(f'{message["role"].upper()}: {content}')
            report.append('')

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text('\n'.join(report))
    print(f'Wrote {OUTPUT_PATH}')

if __name__ == '__main__':
    main()
