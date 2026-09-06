''' Compare promising conversational post-training datasets for TitusAI. '''

import random
from pathlib import Path
from statistics import mean, median

from datasets import load_dataset
from transformers import AutoTokenizer

TOKENIZER_NAME = 'gpt2'
OUTPUT_PATH = Path('data_investigation/output/posttrain_candidates_report.txt')
SAMPLE_SIZE = 500
SAMPLE_CHARACTER_LIMIT = 1200
RANDOM_SEED = 42

CODE_MARKERS = ('```', 'python', 'javascript', 'function', 'code', 'program', 'algorithm', 'class ')
MATH_MARKERS = ('equation', 'calculate', 'solve', 'theorem', 'proof', 'integral', 'derivative', 'algebra')
CASUAL_MARKERS = ('hello', 'hi ', 'hey', 'how are you', 'thank you', 'thanks', 'today', 'friend', 'feel', 'weekend', 'dinner', 'movie', 'music')

def percentile(values, fraction):
    ''' Return a simple percentile from a numeric list. '''
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * fraction))]

def sample_rows(dataset, count):
    ''' Select deterministic random rows from a dataset. '''
    rng = random.Random(RANDOM_SEED)
    indexes = rng.sample(range(len(dataset)), min(count, len(dataset)))
    return [dataset[index] for index in indexes]

def analyze_conversations(name, dataset, tokenizer):
    ''' Summarize a messages-style conversation dataset. '''
    rows = sample_rows(dataset, SAMPLE_SIZE)
    token_counts = []
    assistant_counts = []
    multi_turn = 0
    system = 0
    code_like = 0
    math_like = 0
    casual_like = 0

    for row in rows:
        messages = row['messages']
        contents = [message['content'] or '' for message in messages]
        user_text = '\n'.join(message['content'] or '' for message in messages if message['role'] == 'user').lower()
        assistant_text = '\n'.join(message['content'] or '' for message in messages if message['role'] == 'assistant')
        text = '\n'.join(contents)
        token_counts.append(len(tokenizer.encode(text, add_special_tokens=False)))
        assistant_counts.append(len(tokenizer.encode(assistant_text, add_special_tokens=False)))

        if len(messages) > 2:
            multi_turn += 1
        if any(message['role'] == 'system' for message in messages):
            system += 1
        if any(marker in user_text for marker in CODE_MARKERS):
            code_like += 1
        if any(marker in user_text for marker in MATH_MARKERS):
            math_like += 1
        if any(marker in user_text for marker in CASUAL_MARKERS):
            casual_like += 1

    sampled = len(rows)
    stats = {
        'name': name,
        'rows': len(dataset),
        'sampled': sampled,
        'mean': mean(token_counts),
        'median': median(token_counts),
        'p95': percentile(token_counts, 0.95),
        'assistant_mean': mean(assistant_counts),
        'fit256': sum(value <= 256 for value in token_counts) / sampled,
        'fit512': sum(value <= 512 for value in token_counts) / sampled,
        'fit1024': sum(value <= 1024 for value in token_counts) / sampled,
        'multi_turn': multi_turn / sampled,
        'system': system / sampled,
        'code': code_like / sampled,
        'math': math_like / sampled,
        'casual': casual_like / sampled,
    }
    return stats, rows

def build_oasst_conversations(dataset):
    ''' Build high-quality English human conversation paths from OASST1. '''
    by_id = {row['message_id']: row for row in dataset}
    conversations = []

    for row in dataset:
        if row['role'] != 'assistant' or row['lang'] != 'en':
            continue
        if row['deleted'] or not row['review_result'] or row['synthetic']:
            continue
        if row['rank'] != 0:
            continue

        path = []
        current = row

        while current is not None:
            if current['lang'] != 'en' or current['deleted'] or not current['review_result'] or current['synthetic']:
                path = []
                break
            role = 'user' if current['role'] == 'prompter' else 'assistant'
            path.append({'role': role, 'content': current['text']})
            parent_id = current['parent_id']
            current = by_id.get(parent_id) if parent_id else None

        path.reverse()
        if len(path) >= 2 and path[-1]['role'] == 'assistant':
            conversations.append({'messages': path})

    return conversations

def append_samples(report, name, rows):
    ''' Add a few representative conversations to the report. '''
    report.append('')
    report.append(f'### {name}')
    positions = sorted(set((0, len(rows) // 4, len(rows) // 2, len(rows) - 1)))

    for position in positions:
        messages = rows[position]['messages']
        report.append(f'Sample {position + 1}')
        for message in messages:
            content = (message['content'] or '').strip()
            if len(content) > SAMPLE_CHARACTER_LIMIT:
                content = content[:SAMPLE_CHARACTER_LIMIT] + ' [...]'
            report.append(f'{message["role"].upper()}: {content}')
        report.append('')

def main():
    ''' Compare the strongest researched candidates and write a report. '''
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print('Loading OpenAssistant/oasst1')
    oasst = load_dataset('OpenAssistant/oasst1', split='train')
    oasst_conversations = build_oasst_conversations(oasst)

    print('Loading HuggingFaceTB/smoltalk everyday-conversations')
    everyday = load_dataset('HuggingFaceTB/smoltalk', 'everyday-conversations', split='train')

    print('Loading HuggingFaceTB/smoltalk systemchats-30k')
    systemchats = load_dataset('HuggingFaceTB/smoltalk', 'systemchats-30k', split='train')

    candidates = [
        ('OASST1 human English top-ranked paths', oasst_conversations),
        ('SmolTalk everyday-conversations', everyday),
        ('SmolTalk systemchats-30k', systemchats),
    ]

    report = []
    report.append('TitusAI post-training candidate investigation')
    report.append('============================================')
    report.append('OASST1 is restricted here to English, human-written, reviewed, non-deleted, non-synthetic paths ending in a rank-0 assistant response.')
    report.append(f'Raw OASST1 train messages: {len(oasst):,}')
    report.append(f'Usable OASST1 conversation paths under that strict filter: {len(oasst_conversations):,}')
    report.append('SmolTalk configs are loaded separately; the full 4.15 GB SmolTalk mixture is not downloaded.')
    report.append('Token counts use GPT-2 BPE on raw message content and exclude future Titus role labels/separators/EOS.')
    report.append('Keyword category rates are rough screening signals, not semantic classifiers.')
    report.append('')
    report.append('Candidate comparison')
    report.append('--------------------')
    report.append('candidate | rows | sampled | mean tokens | median | p95 | assistant mean | <=256 | <=512 | <=1024 | multi-turn | system | code-like | math-like | casual-like')

    sampled_rows = {}

    for name, dataset in candidates:
        print(f'Analyzing {name}: {len(dataset):,} rows')
        stats, rows = analyze_conversations(name, dataset, tokenizer)
        sampled_rows[name] = rows
        report.append(
            f'{name} | {stats["rows"]:,} | {stats["sampled"]} | {stats["mean"]:.1f} | {stats["median"]:.1f} | '
            f'{stats["p95"]:,} | {stats["assistant_mean"]:.1f} | {stats["fit256"]:.1%} | {stats["fit512"]:.1%} | '
            f'{stats["fit1024"]:.1%} | {stats["multi_turn"]:.1%} | {stats["system"]:.1%} | {stats["code"]:.1%} | '
            f'{stats["math"]:.1%} | {stats["casual"]:.1%}'
        )

    report.append('')
    report.append('Representative conversations')
    report.append('----------------------------')

    for name, _ in candidates:
        append_samples(report, name, sampled_rows[name])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text('\n'.join(report))
    print(f'Wrote {OUTPUT_PATH}')

if __name__ == '__main__':
    main()
