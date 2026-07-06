import io
import json
import urllib.request


url = 'https://raw.githubusercontent.com/alexa/Topical-Chat/master/conversations/train.json'

with urllib.request.urlopen(url) as response:
    conversations = json.load(io.TextIOWrapper(response, encoding='utf-8'))

print(f'Conversations: {len(conversations):,}')

for index, (conversation_id, conversation) in enumerate(conversations.items()):
    print()
    print('=' * 100)
    print(f'CONVERSATION: {conversation_id}')
    print('=' * 100)

    for turn in conversation['content']:
        print(turn['message'])

    if index == 2:
        break
