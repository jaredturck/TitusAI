from datasets import load_dataset


dataset = load_dataset(
    'allenai/soda',
    split='train',
    streaming=True,
)

dataset = dataset.shuffle(seed=42, buffer_size=10_000)

for index, record in enumerate(dataset.take(10)):
    print()
    print('=' * 100)
    print(f'CONVERSATION {index + 1}')
    print('=' * 100)

    print(f'Narrative: {record["narrative"]}')
    print(f'Speakers: {record["speakers"]}')
    print()

    for message in record['dialogue']:
        print(message)

    print()
    print('TRAINING FORMAT:')
    print('\n'.join(record['dialogue']))

    input('Press Enter for the next conversation...')
