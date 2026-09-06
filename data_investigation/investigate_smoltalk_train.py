''' Compare structurally suitable post-training sources for TitusAI. '''

import random
from collections import Counter
from pathlib import Path
from statistics import mean, median

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

TOKENIZER_NAME = 'gpt2'
OUTPUT_PATH = Path('data_investigation/output/posttrain_structural_report.txt')
RANDOM_SEED = 42
SAMPLES_PER_SOURCE = 20
SAMPLE_CHARACTER_LIMIT = 1000
MAX_TOKENS = 512
MAX_ASSISTANT_TOKENS = 256


def build_oasst_conversations(dataset):
    ''' Build strict English human OASST1 paths ending in top-ranked responses. '''
    by_id = {row['message_id']: row for row in dataset}
    conversations = []

    for row in dataset:
        if row['role'] != 'assistant' or row['lang'] != 'en':
            continue
        if row['deleted'] or not row['review_result'] or row['synthetic'] or row['rank'] != 0:
            continue

        messages = []
        current = row

        while current is not None:
            if current['lang'] != 'en' or current['deleted'] or not current['review_result'] or current['synthetic']:
                messages = []
                break

            role = 'user' if current['role'] == 'prompter' else 'assistant'
            messages.append({'role': role, 'content': current['text']})
            current = by_id.get(current['parent_id']) if current['parent_id'] else None

        messages.reverse()

        if len(messages) >= 2 and messages[-1]['role'] == 'assistant':
            conversations.append({'messages': messages})

    return conversations


def inspect_messages(tokenizer, messages):
    ''' Return token statistics when a conversation fits the structural screen. '''
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

    if not assistant_lengths:
        return None, 'roles'
    if max(assistant_lengths) > MAX_ASSISTANT_TOKENS:
        return None, 'assistant_length'

    return (token_count, sum(assistant_lengths), len(messages), 'system' in roles), None


def add_sample(samples, messages, accepted, rng):
    ''' Keep a deterministic reservoir sample of accepted conversations. '''
    if len(samples) < SAMPLES_PER_SOURCE:
        samples.append(messages)
        return

    slot = rng.randrange(accepted)
    if slot < SAMPLES_PER_SOURCE:
        samples[slot] = messages


def inspect_dataset(name, dataset, tokenizer, rng, source=None):
    ''' Scan a local dataset source with the structural screen. '''
    rejected = Counter()
    token_counts = []
    assistant_counts = []
    turn_counts = []
    samples = []
    considered = 0
    accepted = 0
    system_conversations = 0

    for index in tqdm(range(len(dataset)), desc=name, unit='rows'):
        row = dataset[index]

        if source is not None and row['source'] != source:
            continue

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
        add_sample(samples, row['messages'], accepted, rng)

    return {
        'name': name,
        'considered': considered,
        'accepted': accepted,
        'system': system_conversations,
        'tokens': token_counts,
        'assistant_tokens': assistant_counts,
        'turns': turn_counts,
        'rejected': rejected,
        'samples': samples,
    }


def inspect_ultrachat(tokenizer, rng):
    ''' Stream UltraChat with the same structural screen. '''
    dataset = load_dataset('HuggingFaceH4/ultrachat_200k', split='train_sft', streaming=True)
    rejected = Counter()
    token_counts = []
    assistant_counts = []
    turn_counts = []
    samples = []
    considered = 0
    accepted = 0
    system_conversations = 0

    for row in tqdm(dataset, total=207_865, desc='UltraChat 200k', unit='rows'):
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
        add_sample(samples, row['messages'], accepted, rng)

    return {
        'name': 'UltraChat 200k train_sft',
        'considered': considered,
        'accepted': accepted,
        'system': system_conversations,
        'tokens': token_counts,
        'assistant_tokens': assistant_counts,
        'turns': turn_counts,
        'rejected': rejected,
        'samples': samples,
    }


def append_summary(report, result):
    ''' Add one source summary to the report. '''
    accepted = result['accepted']
    considered = result['considered']
    report.append(f'### {result["name"]}')
    report.append(f'Considered: {considered:,}')
    report.append(f'Accepted: {accepted:,} ({accepted / considered:.1%})')
    report.append(f'Accepted with system messages: {result["system"]:,} ({result["system"] / accepted:.1%})')
    report.append(f'Mean raw tokens: {mean(result["tokens"]):.1f}')
    report.append(f'Median raw tokens: {median(result["tokens"]):.1f}')
    report.append(f'Mean assistant tokens: {mean(result["assistant_tokens"]):.1f}')
    report.append(f'Mean turns: {mean(result["turns"]):.1f}')
    report.append('Rejected: ' + ', '.join(f'{reason}={count:,}' for reason, count in result['rejected'].most_common()))
    report.append('')


def append_samples(report, result):
    ''' Add random accepted conversations to the report. '''
    report.append(f'### {result["name"]}')

    for number, messages in enumerate(result['samples'], 1):
        report.append('')
        report.append(f'Sample {number}')

        for message in messages:
            content = (message['content'] or '').strip()
            if len(content) > SAMPLE_CHARACTER_LIMIT:
                content = content[:SAMPLE_CHARACTER_LIMIT] + ' [...]'
            report.append(f'{message["role"].upper()}: {content}')

    report.append('')


def main():
    ''' Compare the final candidate pool without topic or style filtering. '''
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.model_max_length = 1_000_000
    rng = random.Random(RANDOM_SEED)

    print('Loading OpenAssistant/oasst1')
    oasst_raw = load_dataset('OpenAssistant/oasst1', split='train')
    oasst = build_oasst_conversations(oasst_raw)

    print('Loading HuggingFaceTB/smoltalk everyday-conversations')
    everyday = load_dataset('HuggingFaceTB/smoltalk', 'everyday-conversations', split='train')

    print('Loading HuggingFaceTB/smoltalk systemchats-30k')
    systemchats = load_dataset('HuggingFaceTB/smoltalk', 'systemchats-30k', split='train')

    print('Loading cached HuggingFaceTB/smol-smoltalk train split')
    smol = load_dataset('HuggingFaceTB/smol-smoltalk', split='train')

    results = [
        inspect_dataset('OASST1 strict human paths', oasst, tokenizer, rng),
        inspect_dataset('SmolTalk everyday-conversations', everyday, tokenizer, rng),
        inspect_dataset('SmolTalk systemchats-30k', systemchats, tokenizer, rng),
        inspect_dataset('Smol-SmolTalk short Magpie', smol, tokenizer, rng, 'smol-magpie-ultra-short'),
        inspect_dataset('Smol-SmolTalk OpenHermes', smol, tokenizer, rng, 'openhermes-50k'),
        inspect_dataset('Smol-SmolTalk explore-instruct-rewrite', smol, tokenizer, rng, 'explore-instruct-rewrite'),
        inspect_ultrachat(tokenizer, rng),
    ]

    report = []
    report.append('TitusAI structural post-training investigation')
    report.append('==============================================')
    report.append(f'Screening: <= {MAX_TOKENS} raw GPT-2 tokens, each assistant turn <= {MAX_ASSISTANT_TOKENS} tokens, nonempty messages, valid roles, and an assistant final turn.')
    report.append('No topics or styles are excluded. System messages are kept and counted.')
    report.append('')
    report.append('Source sizes')
    report.append('------------')

    for result in results:
        append_summary(report, result)

    total = sum(result['accepted'] for result in results)
    report.append(f'Total accepted across all sources before deduplication/capping: {total:,}')
    report.append('')
    report.append('Random surviving examples')
    report.append('-------------------------')
    report.append('')

    for result in results:
        append_samples(report, result)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text('\n'.join(report))
    print(f'Wrote {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
