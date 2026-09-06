''' Inspect Smol-SmolTalk for TitusAI post-training. '''

import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

DATASET_NAME = 'HuggingFaceTB/smol-smoltalk'
DATASET_SPLIT = 'test'
TOKENIZER_NAME = 'gpt2'
TOKENIZER_BATCH_SIZE = 512
SAMPLES_PER_SOURCE = 2
SAMPLE_CHARACTER_LIMIT = 1500
OUTPUT_PATH = Path('data_investigation/output/smol_smoltalk_report.txt')

def describe(values):
    ''' Summarize a numeric list. '''
    values = np.asarray(values)
    return f'mean={values.mean():.1f}, median={np.median(values):.1f}, p90={np.percentile(values, 90):.1f}, p95={np.percentile(values, 95):.1f}, max={values.max():,}'

def main():
    ''' Download the investigation split and write a compact report. '''
    print(f'Loading {DATASET_NAME} ({DATASET_SPLIT})')
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    rng = random.Random(42)

    records = []
    contents = []
    content_refs = []
    source_counts = Counter()
    role_counts = Counter()
    role_patterns = Counter()
    samples = defaultdict(list)
    seen_by_source = Counter()
    empty_messages = 0
    system_conversations = 0
    non_assistant_endings = 0
    repeated_roles = 0

    for index, row in enumerate(tqdm(dataset, desc='Scanning conversations')):
        messages = row['messages']
        source = row['source']
        roles = [message['role'] for message in messages]
        source_counts[source] += 1
        role_counts.update(roles)
        role_patterns[' > '.join(roles)] += 1

        if 'system' in roles:
            system_conversations += 1
        if not roles or roles[-1] != 'assistant':
            non_assistant_endings += 1
        if any(roles[position] == roles[position - 1] for position in range(1, len(roles))):
            repeated_roles += 1

        record = {
            'source': source,
            'turns': len(messages),
            'tokens': 0,
            'user_tokens': 0,
            'assistant_tokens': 0,
            'system_tokens': 0,
        }
        records.append(record)

        for message in messages:
            content = message['content'] or ''
            if not content.strip():
                empty_messages += 1
            contents.append(content)
            content_refs.append((index, message['role']))

        seen_by_source[source] += 1
        if len(samples[source]) < SAMPLES_PER_SOURCE:
            samples[source].append(index)
        else:
            slot = rng.randrange(seen_by_source[source])
            if slot < SAMPLES_PER_SOURCE:
                samples[source][slot] = index

    for start in tqdm(range(0, len(contents), TOKENIZER_BATCH_SIZE), desc='Counting GPT-2 tokens'):
        batch = contents[start:start + TOKENIZER_BATCH_SIZE]
        encoded = tokenizer(batch, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False, return_length=True, verbose=False)

        for length, (record_index, role) in zip(encoded['length'], content_refs[start:start + TOKENIZER_BATCH_SIZE]):
            records[record_index]['tokens'] += length
            key = f'{role}_tokens'
            if key in records[record_index]:
                records[record_index][key] += length

    token_counts = [record['tokens'] for record in records]
    turn_counts = [record['turns'] for record in records]
    user_tokens = [record['user_tokens'] for record in records]
    assistant_tokens = [record['assistant_tokens'] for record in records]
    total_tokens = sum(token_counts)

    report = []
    report.append('Smol-SmolTalk investigation report')
    report.append('=================================')
    report.append(f'Dataset: {DATASET_NAME}')
    report.append(f'Split: {DATASET_SPLIT}')
    report.append(f'Rows: {len(dataset):,}')
    report.append('This uses the smaller held-out split for fast investigation; source proportions may differ from train.')
    report.append('Token counts use raw message content only, excluding role labels, separators, and EOS tokens.')
    report.append('')

    report.append('Overall structure')
    report.append('-----------------')
    report.append(f'Turns per conversation: {describe(turn_counts)}')
    report.append(f'GPT-2 content tokens per conversation: {describe(token_counts)}')
    report.append(f'User tokens per conversation: {describe(user_tokens)}')
    report.append(f'Assistant tokens per conversation: {describe(assistant_tokens)}')
    report.append(f'Total raw content tokens: {total_tokens:,}')
    report.append(f'Conversations <= 256 tokens: {sum(value <= 256 for value in token_counts) / len(token_counts):.1%}')
    report.append(f'Conversations <= 512 tokens: {sum(value <= 512 for value in token_counts) / len(token_counts):.1%}')
    report.append(f'Conversations <= 1024 tokens: {sum(value <= 1024 for value in token_counts) / len(token_counts):.1%}')
    report.append(f'Conversations containing system messages: {system_conversations:,} ({system_conversations / len(dataset):.1%})')
    report.append(f'Conversations not ending with assistant: {non_assistant_endings:,} ({non_assistant_endings / len(dataset):.1%})')
    report.append(f'Conversations with consecutive identical roles: {repeated_roles:,} ({repeated_roles / len(dataset):.1%})')
    report.append(f'Empty messages: {empty_messages:,}')
    report.append('')

    report.append('Role counts')
    report.append('-----------')
    for role, count in role_counts.most_common():
        report.append(f'{role}: {count:,}')
    report.append('')

    report.append('Most common role patterns')
    report.append('-------------------------')
    for pattern, count in role_patterns.most_common(12):
        report.append(f'{count:>7,}  {pattern}')
    report.append('')

    report.append('Source distribution and fit')
    report.append('---------------------------')
    report.append('source | rows | share | mean tokens | median tokens | mean turns | <=1024')
    for source, count in source_counts.most_common():
        source_records = [record for record in records if record['source'] == source]
        source_token_counts = [record['tokens'] for record in source_records]
        source_turn_counts = [record['turns'] for record in source_records]
        fit = sum(value <= 1024 for value in source_token_counts) / count
        report.append(f'{source} | {count:,} | {count / len(dataset):.1%} | {np.mean(source_token_counts):.1f} | {np.median(source_token_counts):.1f} | {np.mean(source_turn_counts):.1f} | {fit:.1%}')
    report.append('')

    report.append('Representative conversations')
    report.append('----------------------------')
    for source in sorted(samples):
        report.append('')
        report.append(f'### {source}')
        for index in samples[source]:
            row = dataset[index]
            record = records[index]
            report.append(f'Row {index:,} | {record["turns"]} turns | {record["tokens"]:,} raw GPT-2 tokens')
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
