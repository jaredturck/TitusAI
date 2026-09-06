''' Inspect filtered candidates for the expanded TitusAI post-training corpus. '''

import random
from collections import Counter
from pathlib import Path
from statistics import mean, median

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

TOKENIZER_NAME = 'gpt2'
OUTPUT_PATH = Path('data_investigation/output/posttrain_expansion_report.txt')
RANDOM_SEED = 42
SAMPLES_PER_SOURCE = 15
SAMPLE_CHARACTER_LIMIT = 1000
MAX_TOKENS = 512
MAX_ASSISTANT_TOKENS = 256

CODE_MARKERS = ('```', 'python', 'javascript', 'write code', 'programming', 'algorithm', 'class ', 'def ')
MATH_MARKERS = ('equation', 'calculate', 'solve for', 'theorem', 'proof', 'integral', 'derivative', 'algebra', 'geometry problem')
ROLEPLAY_MARKERS = ('roleplay', 'role-play', 'pretend you are', 'act as ', 'imagine you are', 'you are a wizard')

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

def inspect_messages(tokenizer, messages, reject_system=True):
    ''' Return token statistics when a conversation passes the proposed filter. '''
    roles = [message['role'] for message in messages]

    if reject_system and 'system' in roles:
        return None, 'system'
    if not roles or roles[-1] != 'assistant':
        return None, 'roles'

    user_text = '\n'.join(message['content'] or '' for message in messages if message['role'] == 'user').lower()

    if any(marker in user_text for marker in CODE_MARKERS):
        return None, 'code'
    if any(marker in user_text for marker in MATH_MARKERS):
        return None, 'math'
    if any(marker in user_text for marker in ROLEPLAY_MARKERS):
        return None, 'roleplay'

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

    return (token_count, sum(assistant_lengths), len(messages)), None

def inspect_dataset(name, dataset, tokenizer, source=None):
    ''' Scan one candidate source and keep indexes that pass the proposed filter. '''
    accepted = []
    token_counts = []
    assistant_counts = []
    turn_counts = []
    rejected = Counter()

    for index in tqdm(range(len(dataset)), desc=name, unit='rows'):
        row = dataset[index]

        if source is not None and row['source'] != source:
            continue

        stats, reason = inspect_messages(tokenizer, row['messages'])

        if reason:
            rejected[reason] += 1
            continue

        accepted.append(index)
        token_count, assistant_count, turns = stats
        token_counts.append(token_count)
        assistant_counts.append(assistant_count)
        turn_counts.append(turns)

    return {
        'name': name,
        'dataset': dataset,
        'source': source,
        'accepted': accepted,
        'tokens': token_counts,
        'assistant_tokens': assistant_counts,
        'turns': turn_counts,
        'rejected': rejected,
    }

def append_summary(report, result):
    ''' Add source counts and length statistics to the report. '''
    accepted = len(result['accepted'])
    rejected = result['rejected']
    considered = accepted + sum(rejected.values())
    report.append(f'### {result["name"]}')
    report.append(f'Considered: {considered:,}')
    report.append(f'Accepted: {accepted:,} ({accepted / considered:.1%})')

    if accepted:
        report.append(f'Mean raw tokens: {mean(result["tokens"]):.1f}')
        report.append(f'Median raw tokens: {median(result["tokens"]):.1f}')
        report.append(f'Mean assistant tokens: {mean(result["assistant_tokens"]):.1f}')
        report.append(f'Mean turns: {mean(result["turns"]):.1f}')

    report.append('Rejected: ' + ', '.join(f'{reason}={count:,}' for reason, count in rejected.most_common()))
    report.append('')

def append_samples(report, result, rng):
    ''' Add deterministic random samples from a filtered source. '''
    accepted = result['accepted']
    dataset = result['dataset']
    sample_indexes = rng.sample(accepted, min(SAMPLES_PER_SOURCE, len(accepted)))
    report.append(f'### {result["name"]}')

    for index in sample_indexes:
        row = dataset[index]
        report.append(f'Row {index:,}')

        for message in row['messages']:
            content = (message['content'] or '').strip()
            if len(content) > SAMPLE_CHARACTER_LIMIT:
                content = content[:SAMPLE_CHARACTER_LIMIT] + ' [...]'
            report.append(f'{message["role"].upper()}: {content}')

        report.append('')

def main():
    ''' Build a concrete report for expanding TitusAI post-training to tens of thousands of examples. '''
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.model_max_length = 1_000_000
    rng = random.Random(RANDOM_SEED)

    print('Loading OpenAssistant/oasst1')
    oasst_raw = load_dataset('OpenAssistant/oasst1', split='train')
    oasst = build_oasst_conversations(oasst_raw)

    print('Loading HuggingFaceTB/smoltalk everyday-conversations')
    everyday = load_dataset('HuggingFaceTB/smoltalk', 'everyday-conversations', split='train')

    print('Loading cached HuggingFaceTB/smol-smoltalk train split')
    smol = load_dataset('HuggingFaceTB/smol-smoltalk', split='train')

    results = [
        inspect_dataset('OASST1 strict human paths', oasst, tokenizer),
        inspect_dataset('SmolTalk everyday-conversations', everyday, tokenizer),
        inspect_dataset('Smol-SmolTalk short Magpie', smol, tokenizer, 'smol-magpie-ultra-short'),
        inspect_dataset('Smol-SmolTalk OpenHermes', smol, tokenizer, 'openhermes-50k'),
        inspect_dataset('Smol-SmolTalk explore-instruct-rewrite', smol, tokenizer, 'explore-instruct-rewrite'),
    ]

    report = []
    report.append('TitusAI expanded post-training investigation')
    report.append('===========================================')
    report.append(f'Proposed screening: <= {MAX_TOKENS} raw GPT-2 tokens, each assistant turn <= {MAX_ASSISTANT_TOKENS} tokens, no system messages, and no obvious code, advanced-math, or roleplay prompts.')
    report.append('These are screening heuristics for choosing the final mixture, not the final prepare_data.py implementation.')
    report.append('')
    report.append('Filtered source sizes')
    report.append('---------------------')

    for result in results:
        append_summary(report, result)

    total = sum(len(result['accepted']) for result in results)
    report.append(f'Total accepted across all sources before deduplication/capping: {total:,}')
    report.append('')
    report.append('Random surviving examples')
    report.append('-------------------------')
    report.append('')

    for result in results:
        append_samples(report, result, rng)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text('\n'.join(report))
    print(f'Wrote {OUTPUT_PATH}')

if __name__ == '__main__':
    main()
