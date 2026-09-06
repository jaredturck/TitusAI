''' Inspect UltraChat 200k as a TitusAI post-training expansion candidate. '''

import random
from collections import Counter
from pathlib import Path
from statistics import mean, median

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

TOKENIZER_NAME = 'gpt2'
OUTPUT_PATH = Path('data_investigation/output/posttrain_ultrachat_report.txt')
RANDOM_SEED = 42
SAMPLE_COUNT = 50
SAMPLE_CHARACTER_LIMIT = 1000
MAX_TOKENS = 512
MAX_ASSISTANT_TOKENS = 256


def inspect_messages(tokenizer, messages):
    ''' Return token statistics when an UltraChat conversation fits Titus. '''
    roles = [message['role'] for message in messages]

    if not roles or roles[-1] != 'assistant':
        return None, 'roles'
    if any(role not in ('system', 'user', 'assistant') for role in roles):
        return None, 'roles'

    contents = [(message['content'] or '').strip() for message in messages]

    if any(not content for content in contents):
        return None, 'empty'

    token_count = len(tokenizer.encode('\n'.join(contents), add_special_tokens=False))

    if token_count > MAX_TOKENS:
        return None, 'length'

    assistant_lengths = [
        len(tokenizer.encode(message['content'], add_special_tokens=False))
        for message in messages if message['role'] == 'assistant'
    ]

    if max(assistant_lengths) > MAX_ASSISTANT_TOKENS:
        return None, 'assistant_length'

    return (token_count, sum(assistant_lengths), len(messages), 'system' in roles), None


def main():
    ''' Scan UltraChat train_sft and write exact survivor counts and random samples. '''
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.model_max_length = 1_000_000
    dataset = load_dataset('HuggingFaceH4/ultrachat_200k', split='train_sft', streaming=True)
    rng = random.Random(RANDOM_SEED)
    rejected = Counter()
    token_counts = []
    assistant_counts = []
    turn_counts = []
    samples = []
    considered = 0
    accepted = 0
    system_conversations = 0

    print('Streaming HuggingFaceH4/ultrachat_200k train_sft')

    for row in tqdm(dataset, total=207_865, desc='UltraChat', unit='rows'):
        considered += 1
        stats, reason = inspect_messages(tokenizer, row['messages'])

        if reason:
            rejected[reason] += 1
            continue

        accepted += 1
        token_count, assistant_count, turns, has_system = stats
        token_counts.append(token_count)
        assistant_counts.append(assistant_count)
        turn_counts.append(turns)
        system_conversations += has_system

        if len(samples) < SAMPLE_COUNT:
            samples.append(row['messages'])
        else:
            slot = rng.randrange(accepted)
            if slot < SAMPLE_COUNT:
                samples[slot] = row['messages']

    report = []
    report.append('TitusAI UltraChat post-training investigation')
    report.append('============================================')
    report.append(f'Screening: <= {MAX_TOKENS} raw GPT-2 tokens, each assistant turn <= {MAX_ASSISTANT_TOKENS} tokens, nonempty messages, valid conversation roles, and an assistant final turn.')
    report.append('No topics or styles are excluded. System messages are kept and counted rather than filtered.')
    report.append('UltraChat 200k train_sft is streamed rather than permanently downloading the full dataset.')
    report.append('')
    report.append(f'Considered: {considered:,}')
    report.append(f'Accepted: {accepted:,} ({accepted / considered:.1%})')
    report.append(f'Accepted with system messages: {system_conversations:,} ({system_conversations / accepted:.1%})')
    report.append(f'Mean raw tokens: {mean(token_counts):.1f}')
    report.append(f'Median raw tokens: {median(token_counts):.1f}')
    report.append(f'Mean assistant tokens: {mean(assistant_counts):.1f}')
    report.append(f'Mean turns: {mean(turn_counts):.1f}')
    report.append('Rejected: ' + ', '.join(f'{reason}={count:,}' for reason, count in rejected.most_common()))
    report.append('')
    report.append('Random surviving examples')
    report.append('-------------------------')

    for number, messages in enumerate(samples, 1):
        report.append('')
        report.append(f'Sample {number}')

        for message in messages:
            content = (message['content'] or '').strip()
            if len(content) > SAMPLE_CHARACTER_LIMIT:
                content = content[:SAMPLE_CHARACTER_LIMIT] + ' [...]'
            report.append(f'{message["role"].upper()}: {content}')

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text('\n'.join(report))
    print(f'Wrote {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
