''' Inspect the cached Smol-SmolTalk training split for TitusAI post-training. '''

import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from datasets import load_dataset
from transformers import AutoTokenizer

DATASET_NAME = 'HuggingFaceTB/smol-smoltalk'
TOKENIZER_NAME = 'gpt2'
OUTPUT_PATH = Path('data_investigation/output/smol_smoltalk_train_report.txt')
SAMPLE_SIZE = 200
SAMPLE_CHARACTER_LIMIT = 1200
RANDOM_SEED = 42

CODE_MARKERS = ('```', 'python', 'function', 'code', 'program', 'algorithm')
CONSTRAINT_MARKERS = ('exactly', 'must contain', 'response should', 'bullet points', 'all lowercase', 'all uppercase')

def main():
    ''' Analyze the locally cached train split and write a compact report. '''
    print(f'Loading cached {DATASET_NAME} train split')
    dataset = load_dataset(DATASET_NAME, split='train')
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    rng = random.Random(RANDOM_SEED)

    source_counts = Counter(dataset['source'])
    sampled_indexes = defaultdict(list)
    seen = Counter()

    for index, source in enumerate(dataset['source']):
        seen[source] += 1
        samples = sampled_indexes[source]

        if len(samples) < SAMPLE_SIZE:
            samples.append(index)
        else:
            slot = rng.randrange(seen[source])
            if slot < SAMPLE_SIZE:
                samples[slot] = index

    report = []
    report.append('Smol-SmolTalk training split investigation')
    report.append('=========================================')
    report.append(f'Dataset: {DATASET_NAME}')
    report.append(f'Train rows: {len(dataset):,}')
    report.append(f'Sampled rows per source: up to {SAMPLE_SIZE}')
    report.append('The train split is loaded through datasets; previously downloaded Hugging Face cache files are reused.')
    report.append('Token counts use raw message content only and exclude role labels, separators, and EOS tokens.')
    report.append('')
    report.append('Source counts and sampled characteristics')
    report.append('-----------------------------------------')
    report.append('source | rows | share | sampled | mean tokens | median | <=1024 | multi-turn | system | code-like | constraint-like')

    detailed = {}

    for source, count in source_counts.most_common():
        indexes = sampled_indexes[source]
        rows = [dataset[index] for index in indexes]
        token_counts = []
        multi_turn = 0
        system = 0
        code_like = 0
        constraint_like = 0
        role_patterns = Counter()

        print(f'Inspecting {source}: {count:,} rows, {len(rows)} sampled')

        for row in rows:
            messages = row['messages']
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
            f'{source} | {count:,} | {count / len(dataset):.1%} | {sampled} | {mean(token_counts):.1f} | '
            f'{median(token_counts):.1f} | {fit:.1%} | {multi_turn / sampled:.1%} | {system / sampled:.1%} | '
            f'{code_like / sampled:.1%} | {constraint_like / sampled:.1%}'
        )
        detailed[source] = (indexes, role_patterns)

    report.append('')
    report.append('Representative training conversations')
    report.append('-------------------------------------')

    for source, _ in source_counts.most_common():
        indexes, role_patterns = detailed[source]
        report.append('')
        report.append(f'### {source}')
        report.append('Sampled role patterns: ' + ', '.join(f'{pattern} ({count})' for pattern, count in role_patterns.most_common(4)))

        sample_positions = sorted(set((0, len(indexes) // 2, len(indexes) - 1)))
        for position in sample_positions:
            index = indexes[position]
            row = dataset[index]
            report.append(f'Row {index:,}')
            for message in row['messages']:
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
