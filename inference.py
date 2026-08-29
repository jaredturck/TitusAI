''' Generate responses from the trained language model. '''

from pathlib import Path

import torch
from transformers import AutoTokenizer

from model import LanguageModel

WEIGHTS_PATH = Path('weights')
CONTEXT_LENGTH = 1024
TOKENIZER_NAME = 'gpt2'
MAX_NEW_TOKENS = 512
DEVICE = 'cuda'

checkpoints = sorted(WEIGHTS_PATH.glob('posttrain_*.pt'), key=lambda path: path.stat().st_mtime)

if not checkpoints:
    checkpoints = sorted(WEIGHTS_PATH.glob('model_*.pt'), key=lambda path: path.stat().st_mtime)

model_path = checkpoints[-1]
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = LanguageModel().to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

def generate(prompt):
    text = f'User:\n{prompt}\nAssistant:\n<think>\n'
    tokens = tokenizer(text, return_tensors='pt')['input_ids'].to(DEVICE)
    prompt_length = tokens.shape[1]

    with torch.inference_mode():
        for _ in range(MAX_NEW_TOKENS):
            logits = model(tokens[:, -CONTEXT_LENGTH:])
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat((tokens, next_token), dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    response = tokenizer.decode(tokens[0, prompt_length:], skip_special_tokens=True)

    if '</think>' in response:
        response = response.split('</think>', 1)[1]

    return response.strip()

print(f'Loaded {model_path}')

while True:
    prompt = input('user> ').strip()

    if prompt.lower() in ('exit', 'quit'):
        break

    if prompt:
        print(generate(prompt))
